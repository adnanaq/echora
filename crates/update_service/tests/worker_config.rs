//! Tests for `update_service` settings, worker options, and the worker runner.

use std::collections::HashMap;
use std::net::TcpListener;
use std::time::Duration;

use anyhow::{Context as _, ensure};
use temporal_workflows::testing::start_local_environment;
use temporal_workflows::{
    DEFAULT_TEMPORAL_NAMESPACE, Provider, REFRESH_TRIGGER_SEARCH_ATTRIBUTE, SpikeInput,
    SpikeWorkflow, UPDATE_SERVICE_TASK_QUEUE, WorkerQueueSpec,
};
use temporalio_client::tonic::Request;
use temporalio_client::{Client, ClientOptions, WorkflowGetResultOptions, WorkflowStartOptions};
use temporalio_common::protos::temporal::api::enums::v1::IndexedValueType;
use temporalio_common::protos::temporal::api::operatorservice::v1::{
    AddSearchAttributesRequest, ListSearchAttributesRequest,
};
use temporalio_common::worker::VersioningBehavior;
use temporalio_sdk::testing::{LocalServer, LocalWorkflowEnvironmentOptions, WorkflowEnvironment};
use update_service::{
    UpdateServiceSettings, build_all_worker_options, list_worker_queue_specs, run_workers,
};

const WORKFLOW_WAIT: Duration = Duration::from_secs(30);

fn build_vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn load_settings(pairs: &[(&str, &str)]) -> anyhow::Result<UpdateServiceSettings> {
    UpdateServiceSettings::from_vars(&build_vars(pairs))
}

fn load_settings_error(pairs: &[(&str, &str)]) -> String {
    match load_settings(pairs) {
        Ok(settings) => panic!("expected settings error for {pairs:?}, loaded {settings:?}"),
        Err(settings_error) => format!("{settings_error:#}"),
    }
}

fn find_free_port() -> anyhow::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

async fn start_bare_environment(port: u16) -> anyhow::Result<WorkflowEnvironment<LocalServer>> {
    let options = LocalWorkflowEnvironmentOptions::builder()
        .client_options(ClientOptions::new(DEFAULT_TEMPORAL_NAMESPACE).build())
        .port(port)
        .build();
    Ok(WorkflowEnvironment::start_local(options).await?)
}

async fn wait_for_refresh_trigger(client: &Client) -> anyhow::Result<i32> {
    tokio::time::timeout(WORKFLOW_WAIT, async {
        loop {
            if let Some(attribute_type) = find_refresh_trigger_type(client).await? {
                return Ok(attribute_type);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .context("RefreshTrigger was never created")?
}

async fn find_refresh_trigger_type(client: &Client) -> anyhow::Result<Option<i32>> {
    let listed_attributes = client
        .connection()
        .operator_service()
        .list_search_attributes(Request::new(ListSearchAttributesRequest {
            namespace: DEFAULT_TEMPORAL_NAMESPACE.to_owned(),
        }))
        .await?
        .into_inner();
    Ok(listed_attributes
        .custom_attributes
        .get(REFRESH_TRIGGER_SEARCH_ATTRIBUTE)
        .copied())
}

#[test]
fn settings_from_vars_no_vars_uses_defaults() -> anyhow::Result<()> {
    let settings = load_settings(&[])?;
    assert_eq!(
        settings.connection.target.as_str(),
        "http://localhost:7233/"
    );
    assert!(settings.connection.tls_options.is_none());
    assert!(settings.connection.api_key.is_none());
    assert_eq!(settings.namespace, "echora-dev");
    assert_eq!(
        settings.deployment_version.deployment_name,
        "update-service"
    );
    assert_eq!(
        settings.deployment_version.build_id,
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(settings.versioning_behavior, None);
    Ok(())
}

#[test]
fn settings_from_vars_overrides_replace_defaults() -> anyhow::Result<()> {
    let settings = load_settings(&[
        ("TEMPORAL_ADDRESS", "temporal:7233"),
        ("TEMPORAL_NAMESPACE", "echora"),
        ("WORKER_DEPLOYMENT_NAME", "custom-worker"),
        ("WORKER_BUILD_ID", "build-42"),
        ("WORKER_VERSIONING_ENABLED", "true"),
    ])?;
    assert_eq!(settings.connection.target.as_str(), "http://temporal:7233/");
    assert_eq!(settings.namespace, "echora");
    assert_eq!(settings.deployment_version.deployment_name, "custom-worker");
    assert_eq!(settings.deployment_version.build_id, "build-42");
    assert_eq!(
        settings.versioning_behavior,
        Some(VersioningBehavior::Pinned)
    );
    Ok(())
}

#[test]
fn settings_from_vars_api_key_enables_tls_and_https() -> anyhow::Result<()> {
    let settings = load_settings(&[
        ("TEMPORAL_ADDRESS", "echora.tmprl.cloud:7233"),
        ("TEMPORAL_API_KEY", "secret-key"),
    ])?;
    assert_eq!(
        settings.connection.target.as_str(),
        "https://echora.tmprl.cloud:7233/"
    );
    assert!(settings.connection.tls_options.is_some());
    assert_eq!(settings.connection.api_key.as_deref(), Some("secret-key"));
    Ok(())
}

#[test]
fn settings_from_vars_tls_flag_selects_scheme() -> anyhow::Result<()> {
    let tls_on = load_settings(&[
        ("TEMPORAL_ADDRESS", "temporal:7233"),
        ("TEMPORAL_TLS", "true"),
    ])?;
    assert_eq!(tls_on.connection.target.as_str(), "https://temporal:7233/");
    assert!(tls_on.connection.tls_options.is_some());

    let tls_off_with_key = load_settings(&[
        ("TEMPORAL_ADDRESS", "temporal:7233"),
        ("TEMPORAL_API_KEY", "secret-key"),
        ("TEMPORAL_TLS", "false"),
    ])?;
    assert_eq!(
        tls_off_with_key.connection.target.as_str(),
        "http://temporal:7233/"
    );
    assert!(tls_off_with_key.connection.tls_options.is_none());
    Ok(())
}

#[test]
fn settings_from_vars_explicit_scheme_kept() -> anyhow::Result<()> {
    let settings = load_settings(&[("TEMPORAL_ADDRESS", "https://temporal.internal:7233")])?;
    assert_eq!(
        settings.connection.target.as_str(),
        "https://temporal.internal:7233/"
    );
    Ok(())
}

#[test]
fn settings_from_vars_invalid_address_fails() {
    for address in ["", "http://", "http://[::1", "temporal:7233/path"] {
        let settings_error = load_settings_error(&[("TEMPORAL_ADDRESS", address)]);
        assert!(
            settings_error.contains("invalid Temporal connection settings"),
            "address {address:?} gave {settings_error}"
        );
    }
}

#[test]
fn settings_from_vars_client_certificate_without_key_fails() {
    let settings_error = load_settings_error(&[
        ("TEMPORAL_TLS", "true"),
        ("TEMPORAL_TLS_CLIENT_CERT_DATA", "certificate"),
    ]);
    assert!(settings_error.contains("client key"), "{settings_error}");
}

#[test]
fn settings_from_vars_blank_values_use_defaults() -> anyhow::Result<()> {
    let settings = load_settings(&[
        ("TEMPORAL_NAMESPACE", "  "),
        ("WORKER_DEPLOYMENT_NAME", " "),
        ("WORKER_BUILD_ID", ""),
        ("WORKER_VERSIONING_ENABLED", " "),
        ("WORKER_VERSIONING_BEHAVIOR", ""),
    ])?;
    assert_eq!(settings.namespace, "echora-dev");
    assert_eq!(
        settings.deployment_version.deployment_name,
        "update-service"
    );
    assert_eq!(
        settings.deployment_version.build_id,
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(settings.versioning_behavior, None);
    Ok(())
}

#[test]
fn settings_from_vars_versioning_flag_spellings_parse() -> anyhow::Result<()> {
    for enabled_value in ["true", "TRUE", " True ", "1", "yes", "YES"] {
        let settings = load_settings(&[
            ("WORKER_VERSIONING_ENABLED", enabled_value),
            ("WORKER_BUILD_ID", "build-1"),
        ])?;
        assert_eq!(
            settings.versioning_behavior,
            Some(VersioningBehavior::Pinned),
            "{enabled_value:?} must enable versioning"
        );
    }
    for disabled_value in ["false", "FALSE", "0", "no", "No"] {
        let settings = load_settings(&[("WORKER_VERSIONING_ENABLED", disabled_value)])?;
        assert_eq!(
            settings.versioning_behavior, None,
            "{disabled_value:?} must disable versioning"
        );
    }
    Ok(())
}

#[test]
fn settings_from_vars_invalid_versioning_flag_fails() {
    for flag_value in ["tru", "enabled", "on", "2"] {
        let settings_error = load_settings_error(&[
            ("WORKER_VERSIONING_ENABLED", flag_value),
            ("WORKER_BUILD_ID", "build-1"),
        ]);
        assert!(
            settings_error.contains("WORKER_VERSIONING_ENABLED"),
            "{settings_error}"
        );
    }
}

#[test]
fn settings_from_vars_behavior_spellings_parse() -> anyhow::Result<()> {
    for (behavior_value, expected_behavior) in [
        ("pinned", VersioningBehavior::Pinned),
        ("PINNED", VersioningBehavior::Pinned),
        ("auto_upgrade", VersioningBehavior::AutoUpgrade),
        ("auto-upgrade", VersioningBehavior::AutoUpgrade),
        ("Auto_Upgrade", VersioningBehavior::AutoUpgrade),
    ] {
        let settings = load_settings(&[
            ("WORKER_VERSIONING_ENABLED", "true"),
            ("WORKER_VERSIONING_BEHAVIOR", behavior_value),
            ("WORKER_BUILD_ID", "build-1"),
        ])?;
        assert_eq!(settings.versioning_behavior, Some(expected_behavior));
    }
    Ok(())
}

#[test]
fn settings_from_vars_invalid_behavior_fails() {
    for behavior_value in ["sometimes", "unspecified", "auto"] {
        let settings_error = load_settings_error(&[
            ("WORKER_VERSIONING_ENABLED", "true"),
            ("WORKER_VERSIONING_BEHAVIOR", behavior_value),
            ("WORKER_BUILD_ID", "build-1"),
        ]);
        assert!(
            settings_error.contains("invalid WORKER_VERSIONING_BEHAVIOR"),
            "{settings_error}"
        );
    }
}

#[test]
fn settings_from_vars_behavior_without_versioning_fails() {
    for versioning_pairs in [
        vec![("WORKER_VERSIONING_BEHAVIOR", "pinned")],
        vec![
            ("WORKER_VERSIONING_ENABLED", "false"),
            ("WORKER_VERSIONING_BEHAVIOR", "auto_upgrade"),
        ],
    ] {
        let settings_error = load_settings_error(&versioning_pairs);
        assert!(
            settings_error.contains("WORKER_VERSIONING_ENABLED is not true"),
            "{settings_error}"
        );
    }
}

#[test]
fn settings_from_vars_versioning_without_build_id_fails() {
    for build_id_value in [None, Some(""), Some("  ")] {
        let mut versioning_pairs = vec![("WORKER_VERSIONING_ENABLED", "true")];
        versioning_pairs.extend(build_id_value.map(|build_id| ("WORKER_BUILD_ID", build_id)));
        let settings_error = load_settings_error(&versioning_pairs);
        assert!(
            settings_error.contains("WORKER_BUILD_ID must be set"),
            "{settings_error}"
        );
    }
}

#[test]
fn settings_debug_api_key_redacted() -> anyhow::Result<()> {
    let settings = load_settings(&[
        ("TEMPORAL_ADDRESS", "echora.tmprl.cloud:7233"),
        ("TEMPORAL_API_KEY", "552d3454-secret-api-key"),
    ])?;
    let rendered_settings = format!("{settings:?}");
    assert!(rendered_settings.contains("UpdateServiceSettings"));
    assert!(rendered_settings.contains("<redacted>"));
    assert!(rendered_settings.contains("echora.tmprl.cloud"));
    assert!(!rendered_settings.contains("552d3454-secret-api-key"));
    Ok(())
}

#[test]
fn list_worker_queue_specs_update_service_then_every_provider() {
    let expected_specs: Vec<WorkerQueueSpec> = std::iter::once(WorkerQueueSpec::UpdateService)
        .chain(Provider::ALL.into_iter().map(WorkerQueueSpec::Provider))
        .collect();
    assert_eq!(list_worker_queue_specs(), expected_specs);
    assert!(!list_worker_queue_specs().contains(&WorkerQueueSpec::AgentService));
}

#[test]
fn build_all_worker_options_one_option_per_queue_with_shared_deployment() -> anyhow::Result<()> {
    for versioning_pairs in [
        vec![],
        vec![
            ("WORKER_VERSIONING_ENABLED", "true"),
            ("WORKER_VERSIONING_BEHAVIOR", "auto_upgrade"),
            ("WORKER_BUILD_ID", "build-7"),
        ],
    ] {
        let settings = load_settings(&versioning_pairs)?;
        let all_options = build_all_worker_options(&settings)?;
        let task_queues: Vec<&str> = all_options
            .iter()
            .map(|options| options.task_queue.as_str())
            .collect();
        let expected_task_queues: Vec<&str> = list_worker_queue_specs()
            .into_iter()
            .map(WorkerQueueSpec::task_queue)
            .collect();
        assert_eq!(task_queues, expected_task_queues);
        for options in &all_options {
            assert_eq!(
                options.deployment_options.version,
                settings.deployment_version
            );
            assert_eq!(
                options.deployment_options.use_worker_versioning,
                settings.versioning_behavior.is_some()
            );
            assert_eq!(
                options.deployment_options.default_versioning_behavior,
                settings.versioning_behavior
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn run_workers_running_workers_complete_workflow_then_stop_cleanly() -> anyhow::Result<()> {
    let port = find_free_port()?;
    let environment = start_local_environment(Some(port)).await?;
    let address = format!("127.0.0.1:{port}");
    let settings = load_settings(&[("TEMPORAL_ADDRESS", address.as_str())])?;
    let client = environment.client().clone();

    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
    let workflow_check = async {
        let check_outcome = tokio::time::timeout(WORKFLOW_WAIT, async {
            let handle = client
                .signal_with_start_workflow(
                    SpikeWorkflow::run,
                    SpikeInput {
                        label: "runner".to_owned(),
                    },
                    SpikeWorkflow::finish,
                    "done".to_owned(),
                    WorkflowStartOptions::new(UPDATE_SERVICE_TASK_QUEUE, "run-workers-check")
                        .build(),
                )
                .await?;
            let workflow_output = handle
                .get_result(WorkflowGetResultOptions::default())
                .await?;
            ensure!(workflow_output == "runner:done");
            Ok(())
        })
        .await
        .context("workers never completed the workflow")
        .and_then(|workflow_outcome| workflow_outcome);
        let _ = shutdown_sender.send(());
        check_outcome
    };
    let (run_outcome, check_outcome) = tokio::join!(
        run_workers(&settings, async {
            drop(shutdown_receiver.await);
        }),
        workflow_check,
    );
    check_outcome?;
    run_outcome?;
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn run_workers_unreachable_server_fails_with_address() -> anyhow::Result<()> {
    let settings = load_settings(&[("TEMPORAL_ADDRESS", "127.0.0.1:1")])?;
    let Err(run_error) = run_workers(&settings, std::future::pending()).await else {
        anyhow::bail!("run_workers must fail without a server");
    };
    let run_message = format!("{run_error:#}");
    assert!(
        run_message.contains("failed to connect to Temporal at http://127.0.0.1:1/"),
        "{run_message}"
    );
    Ok(())
}

#[tokio::test]
async fn run_workers_unknown_namespace_fails_before_polling() -> anyhow::Result<()> {
    let port = find_free_port()?;
    let environment = start_local_environment(Some(port)).await?;
    let address = format!("127.0.0.1:{port}");
    let settings = load_settings(&[
        ("TEMPORAL_ADDRESS", address.as_str()),
        ("TEMPORAL_NAMESPACE", "missing-namespace"),
    ])?;
    let run_outcome = tokio::time::timeout(
        WORKFLOW_WAIT,
        run_workers(&settings, std::future::pending()),
    )
    .await
    .context("run_workers kept running against a missing namespace")?;
    let Err(run_error) = run_outcome else {
        anyhow::bail!("run_workers must fail for a missing namespace");
    };
    let run_message = format!("{run_error:#}");
    assert!(
        run_message.contains("Temporal namespace missing-namespace is not available"),
        "{run_message}"
    );
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn run_workers_missing_search_attribute_creates_keyword_once() -> anyhow::Result<()> {
    let port = find_free_port()?;
    let environment = start_bare_environment(port).await?;
    let address = format!("127.0.0.1:{port}");
    let settings = load_settings(&[("TEMPORAL_ADDRESS", address.as_str())])?;
    let client = environment.client().clone();
    ensure!(find_refresh_trigger_type(&client).await?.is_none());

    for _ in 0..2 {
        let (run_outcome, wait_outcome) = tokio::join!(
            run_workers(&settings, async {
                let _ = wait_for_refresh_trigger(&client).await;
            }),
            wait_for_refresh_trigger(&client),
        );
        run_outcome?;
        ensure!(wait_outcome? == i32::from(IndexedValueType::Keyword));
    }
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn run_workers_refresh_trigger_wrong_type_fails() -> anyhow::Result<()> {
    let port = find_free_port()?;
    let environment = start_bare_environment(port).await?;
    environment
        .client()
        .connection()
        .operator_service()
        .add_search_attributes(Request::new(AddSearchAttributesRequest {
            search_attributes: HashMap::from([(
                REFRESH_TRIGGER_SEARCH_ATTRIBUTE.to_owned(),
                i32::from(IndexedValueType::Text),
            )]),
            namespace: DEFAULT_TEMPORAL_NAMESPACE.to_owned(),
        }))
        .await?;
    let address = format!("127.0.0.1:{port}");
    let settings = load_settings(&[("TEMPORAL_ADDRESS", address.as_str())])?;

    let Err(run_error) = run_workers(&settings, std::future::pending()).await else {
        anyhow::bail!("run_workers must fail when RefreshTrigger is not a keyword");
    };
    let run_message = format!("{run_error:#}");
    assert!(run_message.contains("expected Keyword"), "{run_message}");
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn run_workers_shutdown_during_startup_returns_promptly() -> anyhow::Result<()> {
    let silent_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = format!("127.0.0.1:{}", silent_listener.local_addr()?.port());
    let held_connections = tokio::spawn(async move {
        let mut open_sockets = Vec::new();
        while let Ok((socket, _)) = silent_listener.accept().await {
            open_sockets.push(socket);
        }
    });
    let settings = load_settings(&[("TEMPORAL_ADDRESS", address.as_str())])?;

    let run_outcome = tokio::time::timeout(
        Duration::from_secs(5),
        run_workers(&settings, std::future::ready(())),
    )
    .await
    .context("startup ignored shutdown while Temporal did not answer")?;
    run_outcome?;
    held_connections.abort();
    Ok(())
}
