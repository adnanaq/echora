//! Unit tests for `update_service` settings and worker builder.

use std::collections::HashMap;
use std::time::Duration;

use temporal_workflows::{PROVIDER_QUEUE_NAMES, UpdateServicePorts};
use update_service::{
    UpdateServiceSettings, build_all_worker_options, list_worker_queue_specs, run_workers,
};

#[test]
fn settings_load_defaults_and_env_overrides() {
    let defaults = UpdateServiceSettings::from_lookup(|_| None);
    assert_eq!(defaults.temporal_address, "http://localhost:7233");
    assert_eq!(defaults.temporal_namespace, "echora");
    assert_eq!(defaults.deployment_name, "update-service");
    assert_ne!(defaults.build_id, "");
    assert!(!defaults.use_worker_versioning);

    let from_env_settings = UpdateServiceSettings::from_env();
    assert_ne!(from_env_settings.temporal_namespace, "");

    let env = HashMap::from([
        ("TEMPORAL_ADDRESS", "temporal:7233"),
        ("TEMPORAL_NAMESPACE", "custom-ns"),
        ("WORKER_DEPLOYMENT_NAME", "custom-worker"),
        ("WORKER_BUILD_ID", "build-42"),
        ("WORKER_VERSIONING_ENABLED", "true"),
    ]);
    let loaded = UpdateServiceSettings::from_lookup(|key| env.get(key).copied().map(str::to_owned));
    assert_eq!(loaded.temporal_address, "http://temporal:7233");
    assert_eq!(loaded.temporal_namespace, "custom-ns");
    assert_eq!(loaded.deployment_name, "custom-worker");
    assert_eq!(loaded.build_id, "build-42");
    assert!(loaded.use_worker_versioning);
    assert_eq!(
        loaded.enrichment_fetch_service_address,
        "http://localhost:8002"
    );
    assert_eq!(
        loaded.enrichment_merge_service_address,
        "http://localhost:8002"
    );

    let split_env = HashMap::from([
        ("ENRICHMENT_SERVICE_ADDRESS", "http://enrichment:8002"),
        (
            "ENRICHMENT_FETCH_SERVICE_ADDRESS",
            "http://enrichment-fetch:8002",
        ),
        ("ENRICHMENT_MERGE_SERVICE_ADDRESS", "enrichment-merge:8002"),
    ]);
    let split =
        UpdateServiceSettings::from_lookup(|key| split_env.get(key).copied().map(str::to_owned));
    assert_eq!(split.enrichment_service_address, "http://enrichment:8002");
    assert_eq!(
        split.enrichment_fetch_service_address,
        "http://enrichment-fetch:8002"
    );
    assert_eq!(
        split.enrichment_merge_service_address,
        "http://enrichment-merge:8002"
    );

    let workers = build_all_worker_options(&loaded, &UpdateServicePorts::noop())
        .expect("build all worker options");
    assert_eq!(workers.len(), 2 + PROVIDER_QUEUE_NAMES.len());
    assert!(workers[0].deployment_options.use_worker_versioning);
}

#[test]
fn worker_queue_specs_cover_update_providers_and_agent_queues() {
    let specs = list_worker_queue_specs();
    // 1 update-service + 8 providers + 1 agent-service = 10 queues
    assert_eq!(specs.len(), 2 + PROVIDER_QUEUE_NAMES.len());
}

#[tokio::test]
async fn workers_connect_to_dev_temporal_and_shut_down_cleanly() {
    let settings = UpdateServiceSettings::from_env();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
    });
    let outcome = run_workers(&settings, async {
        let _ = shutdown_rx.await;
    })
    .await;
    assert!(outcome.is_ok(), "run_workers failed: {outcome:?}");
}

#[tokio::test]
async fn run_workers_rejects_invalid_url_and_unreachable_server() {
    let mut bad_url = UpdateServiceSettings::from_lookup(|_| None);
    bad_url.temporal_address = "not-a-valid-url".to_owned();
    let bad_url_err = run_workers(&bad_url, std::future::pending())
        .await
        .expect_err("invalid URL must fail");
    assert!(bad_url_err.to_string().contains("TEMPORAL_ADDRESS"));

    let mut unreachable = UpdateServiceSettings::from_lookup(|_| None);
    unreachable.temporal_address = "http://127.0.0.1:1".to_owned();
    let connect_err = run_workers(&unreachable, std::future::pending())
        .await
        .expect_err("unreachable Temporal server must fail");
    assert!(connect_err.to_string().contains("connect"));
}
