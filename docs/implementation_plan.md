# Implementation Plan

This document breaks down the work required to implement the automated architecture proposed in `system_design_proposal.md`.

## Phase 1: Foundational Setup & Core Workflow

### Task 1.0: Initial System Bootstrap

- **Status:** `To-Do`
- **Goal:** Perform the one-time setup to upload the initial data and populate the vector database.

- **Steps:**
  1.  **Create S3 Bucket:** Provision the S3 bucket (e.g., `anime-vector-service-data`) that will serve as the primary data store.
  2.  **Image Ingestion, Deduplication, and Error Handling (Critical):** Develop and run a one-time script to:
      - Provision a new DynamoDB table, `image-deduplication-map`, with `source_url_hash` (SHA256 of the source image URL) as its primary key and a TTL enabled for `last_attempt` timestamps.
      - Provision an SQS Dead-Letter Queue (DLQ) for persistent image ingestion failures.
      - Provision a CloudFront distribution (managed via Pulumi IaC) with the S3 image bucket as its origin, configured for HTTPS-only access.
      - Iterate through the `enriched_anime_database.json`.
      - For each anime, identify all image CDN URLs (covers, posters, banners, character images).
      - For each URL:
        1.  Calculate `source_url_hash` (SHA256 of the source URL).
        2.  Check `image-deduplication-map` using `source_url_hash` to see if the image has already been processed or failed previously.
        3.  **If `source_url_hash` exists and maps to an S3 path:** Reuse the existing S3 path.
        4.  **If `source_url_hash` exists and indicates `DOWNLOAD_FAILED`:** Skip (or re-attempt if `TTL` has expired and `last_attempt` indicates it\'s time).
        5.  **If `source_url_hash` not found or ready for retry:**
            a. Attempt to download the image with a retry mechanism (e.g., `tenacity` with 3-5 retries, exponential backoff, and a total timeout of 30-60 seconds).
            b. **On successful download:**
            _ Calculate `content_hash` (SHA256 of the image binary content).
            _ Define S3 object key as `images/<content_hash>.<extension>` (extension derived from content type).
            * Upload the image to S3 *only if an object with that content hash doesn\'t already exist*. Ensure AWS SDK\'s retry logic is configured appropriately.
            * Store `source_url_hash -> our_s3_bucket_path` (using `content_hash`) in `image-deduplication-map`.
            _ Store original `source_url` and `content_type` as S3 object metadata.
            c. **On persistent download/upload failure (after all retries):**
            _ Send a structured message to the SQS DLQ containing `anime_id`, `image_url`, `error_message`, `timestamp`, `retry_count`, and `context`. \* Store `source_url_hash -> "DOWNLOAD_FAILED"` in `image-deduplication-map` along with a `last_attempt` timestamp and `TTL`.
      - For `AnimeEntry` objects with partially failed images, store the **new CloudFront URL** (e.g., `https://<cloudfront-domain>/images/<content_hash>.<extension>`) for successful images. For failed images, include the `image_url` and `error_message` within the `AnimeEntry` in MongoDB Atlas as a placeholder.
      - This ensures we own and control all image assets from the start, with robust deduplication, retry logic, and comprehensive error handling. The CloudFront URLs stored in `AnimeEntry` will always be HTTPS-only.
  3.  **Data Source Clarity (`anime-offline-database.json` vs `enriched_anime_database.json`):**
      - **`anime-offline-database.json`:** This external file is the **raw, external source of truth** for basic anime metadata. It is never directly ingested into our production DynamoDB or Qdrant.
      - **`enriched_anime_database.json` (Initial Local Version):** This is a **derived artifact** created locally during initial data preparation. It contains `AnimeEntry` objects after programmatic enrichment and image re-hosting (with CloudFront URLs). This local file is manually uploaded to S3 during initial system bootstrap.
      - **`enriched_anime_database.json` (S3 Version):** Once uploaded to S3 (`s3://<bucket-name>/processed/enriched_anime_database.json`), this becomes the **initial snapshot of our system\'s enriched data**. It serves as the baseline for our weekly sync process and as a full, portable backup.
      - **MongoDB Atlas (`animes` collection):** This collection is the **live, operational source of truth** for our enriched `AnimeEntry` objects after the initial bootstrap and subsequent weekly updates.
  4.  **Upload Enriched Database:** Manually upload the locally-generated `enriched_anime_database.json` (now with our S3 image URLs) to `s3://<bucket-name>/processed/enriched_anime_database.json`.
  5.  **Attach Metadata (Critical):** During the upload, attach a custom metadata tag `x-amz-meta-source-commit-sha` containing the commit hash of the `anime-offline-database` version that was used to generate the file. This provides the initial state for the system.
  6.  **Run Bulk Indexing Job:** Trigger a one-time, manual process to populate the Qdrant database. This will be a "scatter-gather" Lambda pattern to process ~40,000 entries in parallel without hitting timeouts.

### Task 1.0.1: Establish Infrastructure as Code (IaC) Framework

- **Status:** `To-Do`
- **Goal:** Define and implement a robust Infrastructure as Code (IaC) framework for provisioning and managing all cloud resources.
- **Rationale:** IaC ensures that our cloud infrastructure is provisioned, configured, and managed in a repeatable, auditable, and version-controlled manner. This minimizes manual errors, facilitates environment consistency, and supports efficient disaster recovery.
- **Decision:** We will use **Pulumi with Python** as our Infrastructure as Code (IaC) framework.
- **Rationale:** This choice aligns with our team's existing Python expertise, allowing us to use a single language for both application and infrastructure code. It enables powerful abstractions, simplifies testing, and reduces context-switching.

- **Repository Structure:**
  - The Pulumi code for our infrastructure will live in its own dedicated Git repository (e.g., `anime-infra`). This separates infrastructure concerns from application code.

- **Initial Setup Requirements:**
  1.  **Git Repository:** A new, dedicated repository for all Pulumi code.
  2.  **Pulumi Account & CLI:** A Pulumi account for state management and the Pulumi CLI installed in local and CI/CD environments.
  3.  **Cloud Credentials:** Securely configured API keys for AWS, MongoDB Atlas, and Qdrant Cloud, managed via environment variables or CI/CD secrets.

- **Pulumi Providers:**
  - **AWS:** The official `pulumi_aws` provider will be used for all AWS resources.
  - **MongoDB Atlas:** The official `pulumi_mongodbatlas` provider will be used. ([Link](https://www.pulumi.com/registry/packages/mongodbatlas/))
  - **Qdrant Cloud:** The official `pulumi_qdrant_cloud` provider will be used. ([Link](https://www.pulumi.com/registry/packages/qdrant-cloud/))

- **Comprehensive List of Resources to Provision (Repository Structure):**

  To ensure maximum modularity, readability, and separation of concerns, the `anime-infra` repository will be structured as follows. Each file will be responsible for a specific set of resources.

  ```
  anime-infra/
  ├── .github/
  │   └── workflows/
  │       └── cicd.yml         # GitHub Actions workflow for infrastructure
  ├── __main__.py              # Main entry point for the Pulumi program
  ├── Pulumi.yaml              # Project definition file
  ├── Pulumi.dev.yaml          # Configuration for the 'dev' environment/stack
  ├── Pulumi.prod.yaml         # Configuration for the 'production' environment/stack
  ├── requirements.txt         # Python dependencies (pulumi, pulumi_aws, etc.)
  └── components/
      ├── __init__.py
      ├── aws/                     # AWS-specific components
      │   ├── __init__.py
      │   ├── networking.py        # VPC, subnets, security groups, NAT/IGW
      │   ├── s3_buckets.py        # S3 buckets (e.g., for images, pipeline data)
      │   ├── dynamodb_tables.py   # DynamoDB tables (e.g., image deduplication)
      │   ├── ecs_cluster.py       # ECS Cluster definition
      │   ├── ecs_services.py      # Fargate Task Definitions and Services (BFF, Agent)
      │   ├── lambda_functions.py  # All Lambda functions for the enrichment pipeline
      │   ├── step_functions.py    # Step Function state machine definition
      │   ├── secrets_manager.py   # Secrets Manager setup
      │   ├── elasticache.py       # ElastiCache for Redis cluster
      │   └── cloudfront.py        # CloudFront distribution for images
      ├── qdrant/                  # Qdrant Cloud-specific components
      │   ├── __init__.py
      │   └── cluster.py           # Qdrant Cloud cluster and API keys
      ├── mongo/                   # MongoDB Atlas-specific components
      │   ├── __init__.py
      │   ├── cluster.py           # MongoDB Atlas cluster
      │   ├── users.py             # Database users for BFF and Agent
      │   ├── network_access.py    # IP Access List / VPC Peering for Atlas
      │   └── collections_indexes.py # MongoDB collections and their indexes
      └── application/             # Application-level wiring of components
          ├── __init__.py
          ├── bff.py               # Wires up AWS ECS service for BFF, connects to Mongo
          └── agent.py             # Wires up AWS ECS service for Agent, connects to Qdrant/Mongo
  ```

  - **How it Works:**
    - **`components/aws/`, `components/qdrant/`, `components/mongo/`:** Each of these subdirectories groups all resources related to a specific cloud provider. Within these, individual files define specific resource types (e.g., `s3_buckets.py` for all S3 buckets).
    - **`components/application/`:** This directory acts as an abstraction layer. For example, `bff.py` will define the *entire BFF application stack* by importing and configuring the necessary AWS ECS services from `aws/ecs_services.py` and connecting them to the MongoDB cluster defined in `mongo/cluster.py`.
    - **`__main__.py`:** This top-level file remains simple. It will primarily import and instantiate the high-level application components from `components/application/`.

- **CI/CD Integration:**
  - Integrate IaC deployments into a CI/CD pipeline using **GitHub Actions** in the new infrastructure repository.
  - **CI Workflow (on Pull Requests):**
    - Runs `pulumi preview` to show planned infrastructure changes. This output can be posted as a comment on the PR for peer review.
  - **CD Workflow (on Merge to `main`):**
    - Executes `pulumi up` to apply infrastructure changes.
    - This job will target protected GitHub Environments (e.g., `staging`, `production`) that require manual approval before deployment.
  - **Authentication:**
    - **AWS:** Utilize OpenID Connect (OIDC) for secure, credential-less authentication from GitHub Actions to AWS IAM.
    - **MongoDB Atlas & Qdrant Cloud:** API keys will be stored as GitHub Secrets and accessed by the Pulumi workflow.
  - **Pulumi State:** The Pulumi Service backend will be used for secure and collaborative state management.

### Task 1.1: Automated Weekly Database Sync

- **Status:** `To-Do`
- **Component:** `Weekly-Sync-Starter-Lambda`
- **Trigger:** AWS EventBridge, scheduled for every Saturday at 01:00 UTC.

- **Logic (Intelligent Sync):**
  1.  Uses the commit SHA in the S3 object metadata to check if the `anime-offline-database` has been updated.
  2.  If so, it downloads the new offline DB.
  3.  It iterates through each anime in the new offline DB and checks for its existence in the main `animes` collection in MongoDB Atlas.
  4.  **For New Anime (not found in the main collection):**
      - **Acquire Lock:** It attempts to acquire a lock by inserting an item into the `anime-processing-locks` collection in MongoDB Atlas using a unique index and write concern to ensure atomicity.
      - **On Success:** If the lock is acquired successfully, it triggers the full `Enrichment-Step-Function` to add the new anime. The Step Function will be responsible for deleting the lock upon completion.
      - **On Failure:** If the lock acquisition fails (meaning another process has already claimed this anime), it does nothing and moves to the next anime.
  5.  **For Existing Anime (found in MongoDB Atlas):**
      - If the entry is currently marked with `system_status == "ORPHANED"`, it will be "resurrected" by setting its status back to `ACTIVE` and clearing the `orphaned_at` timestamp.
      - Performs a selective "diff-and-merge" against the record fetched from MongoDB Atlas to avoid data regression:
      - **Check `status`:** If the `status` in the offline file differs from our record, flag the anime for human review in the validation queue.
      - **Check `episodes` (count):** Compare the integer `episodes` count from the offline file with the actual number of episodes stored in our `episodes` collection (linked by `anime_id`).
        - If `offline_count > our_count`, flag the anime for human review, as we may be missing episodes.
        - If `offline_count <= our_count`, **do nothing**, as our live data is considered more accurate.
      - All other fields from the offline DB for existing entries are ignored to protect our enriched data.
  6.  **For Removed Anime (Orphaning):**
      - After processing the source file, the sync process will identify all anime present in our `animes` collection that were not present in the source file.
      - In line with our goal of creating a comprehensive database, these entries will **never be deleted**.
      - Instead, they will be marked as "orphaned" by updating the entry to include a `system_status: "ORPHANED"` field and an `orphaned_at` timestamp.
      - The API layer will be responsible for filtering these orphaned records from default user-facing queries, but they can be made accessible via a specific query parameter (e.g., `?include_orphaned=true`).

### Task 1.2: Provision Qdrant Vector Database (Qdrant Cloud)

- **Status:** `To-Do`
- **Goal:** To provision a managed, production-ready vector database using Qdrant Cloud.
- **Rationale:** Using a managed service like Qdrant Cloud eliminates the operational burden of self-hosting, including setup, scaling, high availability, and maintenance. This allows the team to focus on application development.

- **Steps:**
  1. **Create Qdrant Cloud Account:** Sign up for a Qdrant Cloud account.
  2. **Provision Cluster:** Create a new vector database cluster through the Qdrant Cloud dashboard. For initial development, the free tier can be used. This can be scaled up to a larger, paid cluster as needed for production.
  3. **Obtain Credentials:** From the cluster dashboard, copy the public **Cluster URL** and generate an **API Key**.
  4. **Configure Application:** Store the Cluster URL and API Key securely (e.g., using AWS Secrets Manager). Update the application configuration so that the `QdrantClient` connects to the cloud endpoint using these credentials.

- **Note on Modularity and Future Alternatives:** The application's vector database client will be implemented via a dedicated adapter module. While Qdrant Cloud is the initial choice, this modular design allows for switching to other managed vector databases in the future with minimal changes to the core application logic. Alternatives like Zilliz Cloud (for Milvus) should be periodically re-evaluated to ensure the chosen provider continues to meet the project's cost and performance needs.

### Task 1.2.1: Provision Enriched Data Store (MongoDB Atlas)

- **Status:** `To-Do`
- **Goal:** To deploy a managed NoSQL document database (MongoDB Atlas) to store enriched `AnimeEntry` objects for fast retrieval and serving user-facing applications. This database will complement the vector database by providing full payload details for anime IDs returned by vector searches.
- **Rationale:** The existing JSON file approach for enriched data lacks the scalability, query flexibility, and operational features required for a production-grade, user-facing data store. MongoDB Atlas offers a flexible document model, rich querying capabilities, and seamless integration with various cloud ecosystems, making it ideal for our complex `AnimeEntry` objects.
- **Steps:**
  1.  **Create MongoDB Atlas Account & Project:** Sign up for a MongoDB Atlas account and create a new project.
  2.  **Provision Cluster:** Create a new MongoDB Atlas cluster (e.g., an M0 Free Tier for initial development, scaling up to M10+ for production). Choose the appropriate cloud provider (AWS) and region.
  3.  **Configure Network Access:** Set up IP Access List entries or VPC Peering to allow connections from your application's environment (e.g., AWS Fargate, Lambda functions).
  4.  **Create Database and Collections:** Within the cluster, create a database (e.g., `anime_service`) and collections for `animes`, `episodes`, and `characters`.
      - **Primary Key:** The `_id` field in MongoDB will store our ULID-based identifiers (`ani_ULID`, `ep_ULID`, `char_ULID`).
      - **Indexing:** Create appropriate indexes on fields like `anime_id` (for episodes/characters), `system_status`, and other frequently queried fields to ensure efficient retrieval.
  5.  **Define Data Model:** The `AnimeEntry`, `EpisodeDetailEntry`, and `CharacterEntry` Pydantic models (`src/models/anime.py`) will serve as the direct schema for documents stored in these collections. We will use a Python ODM (e.g., Beanie) to map Pydantic models to MongoDB documents.
  6.  **Configure IAM Roles/Policies:** Ensure that the FastAPI service (running on Fargate), Lambda functions (for ingestion/consolidation), and any other relevant services have appropriate network access and credentials to connect to MongoDB Atlas. Store connection strings securely (e.g., using AWS Secrets Manager).
  7.  **Initial Data Load Script:** Develop a one-time script to read the existing `enriched_anime_database.json` from S3 and batch-write all `AnimeEntry` objects (and extract/write `EpisodeDetailEntry` and `CharacterEntry` into their respective collections) into the new MongoDB Atlas collections. This will be part of the initial system bootstrap.

### Task 1.2.2: Provision Idempotency Lock Collection (MongoDB Atlas)

- **Status:** `To-Do`
- **Goal:** To create a dedicated MongoDB Atlas collection to act as a distributed lock, ensuring that enrichment workflows are only triggered once per new anime.
- **Rationale:** This prevents race conditions and duplicate workflow executions caused by Lambda retries, without polluting the primary `animes` collection with placeholder records.
- **Steps:**
  1.  **Create MongoDB Collection:** Create a new, simple MongoDB collection (e.g., `anime-processing-locks`) within your MongoDB Atlas database.
      - **Primary Key:** The `_id` field will store the `anime_id` (string) as the unique identifier for the lock.
      - **TTL (Time-to-Live) Index:** Create a TTL index on a `ttl` attribute (timestamp) to automatically clean up stale locks from failed workflows after a set period (e.g., 24 hours).
  2.  **Configure Access:** Ensure that the Lambda functions (for the weekly sync) have appropriate access to read, write, and delete items in this new collection.

### Task 1.3: Design the Enrichment Step Function

- **Status:** `To-Do`
- **Goal:** Define the state machine in AWS Step Functions that orchestrates the enrichment process.
- **Key Feature:** The workflow will include a single callback task that pauses the entire workflow for a final human validation before committing any data.
- **Flow:**
  1.  **Automated Processing:** A series of parallel states will run the programmatic enrichment and all staging scripts (1-5). **This will include a dedicated step for image ingestion:**
      - For each newly fetched or updated `AnimeEntry`, identify all external image CDN URLs.
      - Download these images.
      - Upload them to our S3 bucket (e.g., `s3://<bucket-name>/images/anime/<anime-id>/<image-hash>.jpg`).
      - Update the `AnimeEntry` object to replace external CDN URLs with our internal S3 URLs.
  2.  **Assemble Entry:** A state (implemented as a Lambda function, e.g., `AssembleEntryLambda`) that combines the outputs of all previous steps into a single, final `AnimeEntry` object. **Crucially, this step performs a strict schema validation by parsing the assembled data against the `AnimeEntry` Pydantic model. If validation fails, the workflow will halt, preventing malformed data from proceeding to human review or database commit.**
  3.  **Pause for Validation:** A single **pause state** that sends the complete `AnimeEntry` object to the validation queue and waits for a human to approve, edit, or reject it.
  4.  **Commit Data:** Upon approval, this state runs. It executes the critical dual write to Qdrant and MongoDB Atlas, governed by a multi-layered reliability strategy to ensure data consistency.
      - It **writes the approved `AnimeEntry` object to the MongoDB Atlas collection**, making it live in the system.
      - It saves a copy of the single, approved `AnimeEntry` object to a dedicated S3 prefix (e.g., `processed/updated-entries/<anime-id>.json`) as a permanent audit log.

      - **Commit Strategy: Ensuring Atomicity with a Saga Pattern**
        To guarantee that the dual write to Qdrant and MongoDB is an "all or nothing" operation, the Step Function will implement a Saga pattern with the following layers of resilience:

        **Implementation: The `CommitDataLambda`**

        This entire commit process will be encapsulated within a single, dedicated AWS Lambda function, triggered by the Step Function.

        - **Name:** `CommitDataLambda`
        - **Trigger:** AWS Step Function, upon successful completion of the "Pause for Validation" step.
        - **Input:** The final, human-approved `AnimeEntry` JSON object.
        - **Core Responsibilities:**
          1.  **Instantiate Clients:** The function will contain the Python code to initialize clients for both Qdrant and MongoDB Atlas using the appropriate credentials from AWS Secrets Manager.
          2.  **Execute Qdrant Upsert:** It will call the `qdrant_client.upsert()` method to create or update the vector embeddings in the Qdrant collection.
          3.  **Execute MongoDB Upsert:** It will use a MongoDB client (e.g., PyMongo) to perform an `update_one` operation with `upsert=True` on the `animes` collection, using the `anime_id` as the filter. This writes the full `AnimeEntry` document.
        - **Reliability:** This Lambda is the component that executes the Saga pattern's logic (retries, compensating actions) to ensure the atomicity of the dual write.

        1.  **Layer 1: Automated Retries:** Each database write operation will be wrapped in a retry policy with exponential backoff (e.g., 5 attempts over 2 minutes). This is a native feature of AWS Step Functions and handles the majority of transient network or service errors.

        2.  **Layer 2: Saga Orchestration:** The commit logic follows a precise sequence to prevent data inconsistency.
            - **Step A: Write to Qdrant.** The vector data is written to Qdrant first. If this write fails after all retries, the entire workflow fails, leaving the primary data store untouched.
            - **Step B: Write to MongoDB Atlas.** If the Qdrant write succeeds, the full `AnimeEntry` object is written to the `animes` collection in MongoDB Atlas.
            - **Step C: Compensating Action (Rollback).** If the MongoDB write fails permanently, the Saga triggers a compensating action to **delete the record that was just created in Qdrant.** This ensures no "orphaned" vectors exist in the search index.

        3.  **Layer 3: Dead-Letter Queue (DLQ) for Human Intervention:** In the exceptional case that the Saga itself fails (e.g., the compensating action fails), the entire Step Function execution, along with the `AnimeEntry` payload, is sent to an SQS Dead-Letter Queue. A CloudWatch Alarm monitoring the DLQ will notify the engineering team, allowing a human to manually investigate the rare failure and ensure data consistency.

  5.  **Cleanup:** A final state to clean up any temporary resources from EFS.

### Task 1.4: Implement Portability & Backup Job

- **Status:** `To-Do`
- **Component:** `create-database-snapshot-lambda`
- **Trigger:** EventBridge (scheduled, e.g., weekly on Sunday).
- **Goal:** To create a periodic, full, cloud-agnostic snapshot of the entire enriched database. This file serves as a crucial artifact for disaster recovery and simplifies potential future migrations to other cloud providers.
- **Logic:**
  1.  The Lambda is triggered by a weekly schedule.
  2.  It connects to MongoDB Atlas and performs an export of all records from the `animes`, `episodes`, and `characters` collections.
  3.  It assembles all the records into a single `enriched_anime_database.json` file (or a set of files, one per collection).
  4.  It writes the complete file(s) to S3, overwriting the previous week's snapshot. This file now represents a complete, portable backup of the database state.

### Task 1.5: Provision Observability Platform (SigNoz)

- **Status:** `To-Do`
- **Goal:** Deploy a dedicated, self-hosted SigNoz instance in AWS to collect and visualize logs, metrics, and traces from all application components.
- **Rationale:** To provide a robust, open-source, and self-hosted observability stack, giving us deep insights into application performance and behavior without vendor lock-in. This is a foundational component for maintaining a production-grade system.
- **Architecture:**
  - **Compute:** A dedicated EC2 instance (e.g., `t3.large` or `t4g.large`) will be provisioned in a private VPC subnet. It will have a persistent EBS volume (e.g., 100 GB `gp3`) for storing telemetry data.
  - **Networking:** A dedicated Security Group will allow inbound traffic from application services (on OTel ports `4317`/`4318`) and from trusted IPs for accessing the web UI (on port `3301`).
  - **DNS:** A private Route 53 DNS record (e.g., `signoz.internal.anime-vector-service`) will be created for stable service discovery.
- **Implementation Steps:**
  1.  **IaC Integration:** Define all AWS resources (EC2, EBS, Security Group, Route 53) within the chosen IaC framework (Pulumi/Terraspace).
  2.  **Automated Deployment:** Use a `cloud-init` script to install Docker/Docker-Compose and launch the SigNoz services on the instance.
  3.  **Application Instrumentation:**
      - Add the OpenTelemetry SDK to the Python application dependencies.
      - Instrument the FastAPI service, Lambda functions, and key scripts.
      - Configure the services via environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT`) to send telemetry data to the SigNoz instance.

#### Detailed Implementation Example

This section provides a more concrete, low-level example of how Task 1.5 would be implemented.

**1. Infrastructure as Code (IaC) Pseudo-code:**

The following demonstrates how the resources could be defined in a framework like Pulumi.

```python
# 1. Define the Security Group for SigNoz
signoz_sg = aws.ec2.SecurityGroup('signoz-sg',
    description='Allow SigNoz traffic',
    vpc_id=vpc.id,
    ingress=[
        # Allow OTLP data from our app's security group (e.g., sg-app-fargate)
        {'protocol': 'tcp', 'from_port': 4317, 'to_port': 4317, 'security_groups': [app_sg.id]},
        # Allow access to the web UI from a trusted IP
        {'protocol': 'tcp', 'from_port': 3301, 'to_port': 3301, 'cidr_blocks': ['YOUR_VPN_OR_OFFICE_IP/32']},
    ]
)

# 2. Define the EC2 instance with a startup script
signoz_instance = aws.ec2.Instance('signoz-instance',
    instance_type='t3.large',
    ami='<latest-amazon-linux-2-ami>',
    vpc_security_group_ids=[signoz_sg.id],
    user_data="""
        #!/bin/bash
        yum update -y
        yum install -y docker git
        systemctl start docker
        pip3 install docker-compose
        git clone https://github.com/SigNoz/signoz.git /opt/signoz
        cd /opt/signoz/deploy/docker/clickhouse-setup
        docker-compose up -d
    """
)

# 3. Create a private DNS record
dns_record = aws.route53.Record('signoz-dns',
    zone_id=private_dns_zone.id,
    name='signoz.internal.anime-vector-service',
    type='A',
    records=[signoz_instance.private_ip]
)
```

**2. Application Instrumentation (FastAPI Example):**

In `src/main.py`, the application would be configured to export telemetry.

- **Dependencies (`pyproject.toml`):**

  ```toml
  opentelemetry-api
  opentelemetry-sdk
  opentelemetry-exporter-otlp
  opentelemetry-instrumentation-fastapi
  opentelemetry-instrumentation-requests
  ```

- **Code (`src/main.py`):**

  ```python
  from fastapi import FastAPI
  from opentelemetry import trace
  from opentelemetry.sdk.resources import Resource
  from opentelemetry.sdk.trace import TracerProvider
  from opentelemetry.sdk.trace.export import BatchSpanProcessor
  from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
  from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
  import os

  # Configure the OTel SDK
  resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "default-service")})
  tracer_provider = TracerProvider(resource=resource)
  trace.set_tracer_provider(tracer_provider)
  otlp_exporter = OTLPSpanExporter() # Endpoint configured by env var
  tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

  # Instrument the app
  app = FastAPI()
  FastAPIInstrumentor.instrument_app(app)
  ```

- **Environment Variables (in Fargate Task Definition):**
  - `OTEL_SERVICE_NAME`: `anime-fastapi-service`
  - `OTEL_EXPORTER_OTLP_ENDPOINT`: `http://signoz.internal.anime-vector-service:4317`

### Task 1.6: Implement Cost Management & Governance

- **Status:** `To-Do`
- **Goal:** To establish a comprehensive strategy for monitoring, controlling, and optimizing AWS costs from the beginning of the project.
- **Rationale:** The number and scale of AWS services used can lead to unexpected costs if not proactively managed. This task ensures visibility and governance are built-in.
- **Implementation Steps:**
  1.  **AWS Budgets & Alerts:**
      - Create AWS Budgets for the overall project and for key services (e.g., Fargate, DynamoDB, A2I).
      - Configure tiered budget alerts at 50% and 90% of the monthly budget.
      - Route alerts via SNS to a dedicated Slack channel (e.g., `#cost-alerts`) for immediate visibility.
  2.  **Programmatic Tag Enforcement:**
      - Define a strict resource tagging policy requiring `project`, `environment`, and `service` tags for all resources.
      - Implement this policy using Pulumi's Policy as Code framework to programmatically prevent the creation of non-compliant resources.
  3.  **Cost Anomaly Detection:**
      - Enable AWS Cost Anomaly Detection to automatically identify unusual spending patterns.
      - Establish a bi-weekly review of detected anomalies to understand spending and refine the detector.
  4.  **Proactive Optimization:**
      - Establish a recurring monthly "Cost Optimization & Right-Sizing Review" task in the project backlog.
      - Use data from the SigNoz observability platform to analyze resource utilization (CPU, memory, etc.) and inform right-sizing decisions for EC2 instances and Fargate tasks.
      - Periodically evaluate DynamoDB capacity modes, moving from On-Demand to Provisioned with auto-scaling as traffic patterns become clear.

### Task 1.7: Establish Local Development Environment

- **Status:** `To-Do`
- **Goal:** To provide a comprehensive, easy-to-use local development environment that closely mimics the cloud infrastructure, enabling high developer productivity.
- **Rationale:** A poor or inconsistent local development experience can significantly slow down the team and introduce bugs. This task aims to create a "one-command setup."
- **Implementation Steps:**
  1.  **Docker Compose Orchestration:**
      - Create a `docker-compose.local.yml` file to define and orchestrate local versions of all key dependencies:
        - `vector-service`: The FastAPI application.
        - `qdrant`: The official Qdrant image.
        - `mongodb`: The official `mongo` Docker image. **This provides a local, API-compatible instance of MongoDB, ensuring consistency with our production MongoDB Atlas data store.**
        - `mongo-express`: (Optional) A web-based administration interface for MongoDB.
        - `minio`: To act as an S3-compatible object store.
        - `signoz`: The full SigNoz observability stack.
  2.  **Configuration Management:**
      - Create a `.env.local.example` file containing all necessary environment variables for the `vector-service` to connect to the other local containers (e.g., `MONGO_URI=mongodb://mongodb:27017/anime_service`).
      - This file will be git-ignored and developers can copy it to `.env.local` for their specific setup.
  3.  **Data Seeding:**
      - Create a `scripts/seed_local_env.py` script.
      - This script will populate the local Qdrant and MongoDB instances with a small, consistent set of sample data, allowing developers to have a working environment immediately after setup. The `AnimeEntry`, `EpisodeDetailEntry`, and `CharacterEntry` Pydantic models (`src/models/anime.py`) will serve as the schema for data stored in the local MongoDB.
  4.  **Documentation:**
      - Create a detailed `README.local.md` explaining how to:
        - Install prerequisites (Docker).
        - Run one command to launch the environment (`docker-compose -f docker-compose.local.yml up`).
        - Run the data seeding script.
        - Set up dummy AWS credentials (e.g., `AWS_ACCESS_KEY_ID=dummy`) required by the AWS SDK.
  5.  **Deferred Tasks:**
      - Local emulation of the `Enrichment-Step-Function` is explicitly deferred for V1. Developers will rely on a shared `dev` AWS environment for end-to-end workflow testing.

## Phase 2: Human-in-the-Loop Validation with Amazon A2I

### Task 2.1: Implement A2I Human Review Workflow

- **Status:** `To-Do`
- **Component:** Amazon Augmented AI (A2I)
- **Goal:** To integrate a managed, secure human review step directly into the `Enrichment-Step-Function`.
- **Logic:**
  1.  **Create a Private Workforce:** Set up a private workforce in Amazon A2I, inviting your team members as reviewers.
  2.  **Define a Human Review Workflow (Flow Definition):** Configure a flow definition that specifies the workforce, the worker template (from Task 2.2), and integrates with AWS Step Functions.
  3.  **Integrate with Step Functions:** Replace the `Pause for Validation` state in the Step Function (Task 1.3) with a native A2I `StartHumanLoop` task. This task will automatically route the `AnimeEntry` object to the A2I workforce and pause the execution until the review is complete.

### Task 2.2: Design and Implement Custom A2I Worker Template

- **Status:** `To-Do`
- **Component:** Amazon A2I Worker Template
- **Goal:** To create a user-friendly web interface within A2I for validators to efficiently review, edit, and approve/reject anime data.
- **Functionality:**
  1.  **Design the Template:** Create a custom HTML template using the Liquid templating language. The template will receive the `AnimeEntry` JSON object as input.
  2.  **Render Editable Form (Tabbed Interface):** The template will render the `AnimeEntry` data using a tabbed interface to manage complexity, especially for large entries. Each tab will focus on a specific logical section of the `AnimeEntry` (e.g., "Core Info," "Characters," "Relationships," "Episodes"). Within each tab, the data will be presented in a structured, editable form, allowing validators to easily review, edit, add, or delete specific fields.
  3.  **Implement Approval/Rejection Logic:** The template will include "Approve" and "Reject" buttons.
  4.  **Handle Data Output:** When a reviewer clicks "Approve", the template will consolidate all (potentially edited) data from the form fields into a final `AnimeEntry` JSON object. This object is the output of the A2I step and is passed back to the Step Function.
  5.  **Resume Logic:** The Step Function will automatically resume upon completion of the human loop. If approved, it will proceed to the `Commit Data` state with the final, human-verified data. If rejected, it can be routed to a failure state.

## Phase 3: Live Episode Update Workflow

### Task 3.1: Implement Daily Dynamic Scheduler

- **Status:** `To-Do`
- **Component:** `daily-scheduler-lambda`
- **Trigger:** EventBridge (runs daily at 00:05 UTC).
- **Goal:** To find all anime airing in the next 24 hours (for ongoing shows) and all anime premiering in the next 24 hours (for upcoming shows), and create precise triggers for them.
- **Logic:**
  1.  **Handle Ongoing Shows:** Queries the `animes` collection in MongoDB Atlas for all entries with `status == "ONGOING"`. For each, it finds the next episode scheduled to air in the next 24 hours and creates a one-time EventBridge schedule to trigger the `run-single-episode-update-lambda` at the target time.
      - **Idempotency:** To prevent duplicate schedules upon retry, a deterministic, unique name will be used for each schedule (e.g., `animeId-{anime_id}-episode-{episode_number}`). An attempt to create a schedule with a name that already exists will be caught and treated as a success.
  2.  **Handle Upcoming Shows:** Queries the `animes` collection in MongoDB Atlas for all entries with `status == "UPCOMING"` and a premiere date within the next 24 hours. For each of these anime, it proactively triggers the full `Enrichment-Step-Function`.
      - **Idempotency:** This process will use the same locking mechanism described in Task 1.1 (using the `anime-processing-locks` collection) to prevent duplicate workflow triggers for the same premiere.

### Task 3.2: Implement Single Episode Update Trigger

- **Status:** `To-Do`
- **Component:** `run-single-episode-update-lambda`
- **Trigger:** Amazon EventBridge Scheduler (dynamically, per episode).
- **Goal:** To kick off the enrichment workflow for a single airing episode.
- **Logic:**
  1.  Receives the `anime_id` and `episode_number` from the EventBridge schedule.
  2.  Triggers the `Enrichment-Step-Function`, passing the `anime_id` and an input that signifies an "episode update" mode.

### Task 3.3: Adapt Staging Scripts for Timezone Correction

- **Status:** `To-Do`
- **Component:** Staging scripts (e.g., `process_stage2_episodes.py`)
- **Goal:** To implement the critical timezone conversions identified in `docs/timezone_analysis.md`.
- **Logic:** The scripts will be updated to ensure that any date/time information from external APIs (especially JST) is correctly converted to UTC before being included in the final `AnimeEntry` object.

## Phase 4: Caching & Performance Optimization (GraphQL Focused)

This phase is updated to reflect the shift to a GraphQL API, which moves the primary caching burden from the edge (API Gateway) to the application layer (the BFF Service).

### Task 4.1: Provision Caching Infrastructure

- **Status:** `To-Do`
- **Component:** Amazon ElastiCache for Redis
- **Goal:** To deploy a managed Redis cluster within the VPC to serve as a high-speed cache for the BFF and other backend services.
- **Configuration:** A cache instance (e.g., `cache.t3.small`) will be provisioned in a private subnet, with a security group that only allows access from the Fargate services (BFF and Agent Service) and relevant Lambda functions.

### Task 4.2: Integrate Caching for External API Calls

- **Status:** `To-Do`
- **Component:** Python data processing scripts and Lambdas.
- **Goal:** To reduce redundant API calls, avoid rate-limiting, and speed up the backend enrichment process.
- **Logic:**
  1.  The core data fetching method in each helper will be modified.
  2.  Before making a live HTTP request, it will first check the Redis cache for the requested data using a standardized key (e.g., `jikan:anime:123`).
  3.  **On a cache hit,** it will return the cached data immediately.
  4.  **On a cache miss,** it will perform the real API request, save the result to the Redis cache with an appropriate TTL (Time-To-Live), and then return the data.

### Task 4.2.1: Integrate Caching for Enriched Data (MongoDB Atlas)

- **Status:** `To-Do`
- **Component:** **BFF Service (Bun/ElysiaJS)**
- **Goal:** To significantly reduce latency and MongoDB Atlas read costs for retrieving `AnimeEntry` objects.
- **Logic:**
  1.  **Caching Mechanism:** Utilize Amazon ElastiCache for Redis as a distributed, in-memory cache.
  2.  **Caching Strategy (Cache-Aside):**
      - **Read Path:** When the BFF service needs to retrieve an `AnimeEntry` from MongoDB Atlas, it will first check the Redis cache using a standardized key (`anime:<anime_id>`).
      - **Cache Hit:** If the `AnimeEntry` is found in Redis, it will be deserialized and returned immediately.
      - **Cache Miss:** If not found, the `AnimeEntry` will be fetched from MongoDB Atlas, stored in Redis with a TTL, and then returned.
  3.  **Cache Invalidation:**
      - When an `AnimeEntry` is **created or updated** by the `Enrichment Step Function`, the corresponding cache entry in Redis will be explicitly **deleted**. This ensures data consistency.
  4.  **Data Loader Pattern:** To prevent the N+1 problem in GraphQL resolvers, the BFF will use a **Data Loader**. This pattern batches multiple requests for individual anime (e.g., when resolving a list) into a single, efficient `find({ _id: { $in: [...] } })` query to both the Redis cache and MongoDB.

### Task 4.3: Re-evaluate API Gateway Caching

- **Status:** `To-Do`
- **Component:** Amazon API Gateway
- **Goal:** To acknowledge the reduced effectiveness of edge caching with GraphQL.
- **Logic:**
  1.  Due to GraphQL sending most requests to a single `/graphql` endpoint, caching based on URL paths is no longer effective for dynamic queries.
  2.  API Gateway caching will be **disabled** for the `/graphql` endpoint.
  3.  It may still be considered for other static assets or potential future REST endpoints served by the BFF, but it is no longer a primary component of the main API caching strategy.

### Task 4.4: Implement Application-Layer Caching for Curated Lists

- **Status:** `To-Do`
- **Component:** BFF Service & a new scheduled Lambda function.
- **Goal:** To provide fast responses for frequently accessed, non-personalized lists like "Top Rated Anime" or "Trending This Season".
- **Logic:**
  1.  A new, scheduled **AWS Lambda function** will run periodically (e.g., every hour).
  2.  This Lambda will perform the expensive aggregation query on MongoDB Atlas to determine the list of "top" or "trending" anime.
  3.  It will write the resulting ordered list of anime IDs to a specific key in **Redis** (e.g., `curated_list:top_rated`).
  4.  When the BFF receives a GraphQL query for this list, it will fetch the list of IDs directly from Redis, and then use the Data Loader (from Task 4.2.1) to efficiently retrieve the full `AnimeEntry` objects. This makes the API response extremely fast.

## Phase 5: API Strategy (BFF & Agent Service Model)

This phase outlines the modern, two-service architecture for the user-facing API. It consists of a public-facing GraphQL BFF (Backend-for-Frontend) and an internal, AI-powered Agent Service. This polyglot (TypeScript + Python) model uses the best technology for each specific job.

### Component 1: The BFF Service (Bun/ElysiaJS + GraphQL)

- **Technology:** Bun/ElysiaJS (TypeScript) and GraphQL.
- **Deployment:** A serverless AWS Fargate container, exposed publicly via Amazon API Gateway.
- **Responsibilities:**
  - Acts as the single gateway for the frontend application.
  - Exposes a comprehensive GraphQL schema for all frontend data requirements.
  - Receives natural language search queries and passes them to the internal Agent Service.
  - Receives search results (anime IDs) from the Agent Service.
  - Fetches full `AnimeEntry` documents from MongoDB Atlas using the IDs.
  - Implements the application-layer caching strategy (see Phase 4) to ensure performance.
  - Handles all data shaping and serves the final GraphQL response to the frontend.

### Component 2: The Agent Service (Python)

- **Technology:** Python, using frameworks like `atomic-agents` to orchestrate LLM interactions.
- **Deployment:** A serverless AWS Fargate container. This service is **internal-only** and is not exposed to the public internet. It will be placed behind an internal Application Load Balancer, allowing the BFF to communicate with it securely and with low latency.
- **Responsibilities:**
  - Exposes a simple, internal REST or gRPC endpoint to receive natural language queries from the BFF.
  - Uses an LLM (e.g., from Amazon Bedrock) to parse the natural language query into a structured search request. This includes generating the appropriate `embedding_text` and structured `filters`.
  - Uses the `QdrantClient` to execute a complex, multi-vector search against the Qdrant database using the parameters provided by the LLM.
  - Returns a ranked list of anime IDs to the BFF.

### High-Level Request Flow (Natural Language Search)

1.  **Frontend -> BFF:** Sends a GraphQL query containing the user's search string.
2.  **BFF -> Agent Service:** Makes an internal API call, passing the raw search string.
3.  **Agent Service -> LLM (Bedrock):** Asks the LLM to convert the string into structured search parameters.
4.  **Agent Service -> Qdrant:** Executes the search and gets a list of anime IDs.
5.  **Agent Service -> BFF:** Returns the list of IDs.
6.  **BFF -> MongoDB:** Fetches the full anime documents for the given IDs.
7.  **BFF -> Frontend:** Returns the final data in the requested GraphQL format.

### Architectural Clarification: Application Services vs. Managed Services

It's important to distinguish between the **application services we build and deploy** and the **managed/third-party services we consume**.

In this architecture, we are building **two** primary application services:

1.  **BFF (Backend-for-Frontend) Service:** Written in TypeScript (Bun/ElysiaJS). Its job is to serve the frontend client, manage user data, and act as a simple gateway. It talks to a **MongoDB Atlas** database.
2.  **Agent Service:** Written in Python. This is the "brains" of the operation. It handles the complex vector search, AI-powered enrichment, and data processing pipelines. It talks to **Qdrant Cloud** for vector storage and an external **LLM API** for AI tasks.

Therefore, while we interact with three major data/AI components (MongoDB, Qdrant, LLM), these are consumed by our two distinct application services. This model provides a clean separation of concerns and leverages the strengths of each technology.

### Detailed Infrastructure Diagram (Top-Down View)

```
                                         +--------------------------------------------------+
                                         |                   USER'S DEVICE                  |
                                         |                  (Web Browser)                   |
                                         +-------------------------+------------------------+
                                                                   |
                                                                   | (HTTPS Requests)
                                                                   |
+------------------------------------------------------------------V---------------------------------------------------------------------+
|                                                                                                                                       |
|                                                      AWS CLOUD ENVIRONMENT                                                              |
|                                                                                                                                       |
|  +-----------------------------------------------------------------------------------------------------------------------------------+  |
|  |                                                                                                                                   |  |
|  |  +---------------------------------+      (Internal API Call)      +----------------------------------+       (API Call)       +------------------+
|  |  |      BFF SERVICE (Fargate)      +-----------------------------> |    AGENT SERVICE (Fargate)       +----------------------> |  LLM API         |
|  |  | (Bun/ElysiaJS - GraphQL API)    |                               | (Python - Internal REST/gRPC API)|                        | (OpenAI/Anthropic) |
|  |  +-----------------+---------------+                               +----------------+-----------------+                        +------------------+
|  |                    |                                                                |
|  | (DB Query)         |                                                                | (Vector Search/Upsert)
|  |                    |                                                                |
|  |  +-----------------V---------------+                               +----------------V-----------------+
|  |  |      MongoDB Atlas              |                               |        Qdrant Cloud              |
|  |  | (Managed Document Database)     |                               | (Managed Vector Database)        |
|  |  +---------------------------------+                               +----------------------------------+
|  |                                                                                      ^
|  |                                                                                      | (Write Enriched Data)
|  |                                                                                      |
|  |  +----------------------------------------------------------------------------------+--------------------------------------------+
|  |  |                                    OFFLINE DATA ENRICHMENT PIPELINE (Serverless)                                             |
|  |  |                                                                                                                               |
|  |  |  +------------------------+      +------------------+      +---------------------+      +---------------------+      +--------------------+
|  |  |  | Enrichment Trigger     |----->| Step Function    |----->|  Lambda: Fetch APIs |----->|  Lambda: Process    |----->| Lambda: Assemble   |
|  |  |  | (e.g., Cron, Manual)   |      | (Orchestrator)   |      | (api_fetcher.py)    |      | (process_stage*.py) |      | & Write to Qdrant  |
|  |  |  +------------------------+      +--------+---------+      +----------+----------+      +----------+----------+      +----------+---------+
|  |  |                                           |                           |                           |                           |
|  |  |                                           +---------------------------+---------------------------+---------------------------+
|  |  |                                                                       |
|  |  |                                                                       | (Read/Write Intermediate Files)
|  |  |                                                                       |
|  |  |                                                         +-------------V-------------+
|  |  |                                                         |      S3 Bucket              |
|  |  |                                                         | (Temp Storage for JSONs)    |
|  |  |                                                         +-----------------------------+
|  |  |                                                                                                                               |
|  |  +-------------------------------------------------------------------------------------------------------------------------------+
|  |                                                                                                                                   |
|  +-----------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                       |
+---------------------------------------------------------------------------------------------------------------------------------------+
```

### Mermaid Diagram

```mermaid
graph TD
    subgraph User
        Client[Web Browser]
    end

    subgraph "External Managed Services"
        MongoDB["MongoDB Atlas (Document DB)"]
        QdrantDB["Qdrant Cloud (Vector DB)"]
        LLM_API["LLM API (OpenAI/Anthropic)"]
    end

    subgraph "AWS Cloud"
        subgraph "Real-time Services (AWS Fargate)"
            BFF["BFF Service (Bun/ElysiaJS)"]
            Agent["Agent Service (Python)"]
        end

        subgraph "Offline Enrichment Pipeline (Serverless)"
            Trigger[Enrichment Trigger]
            StepFunction["Step Function Orchestrator"]
            S3["S3 Bucket (Temp JSON Storage)"]
            LambdaFetch["Lambda: Fetch APIs"]
            LambdaProcess["Lambda: Process Stages"]
            LambdaWrite["Lambda: Assemble & Write"]
        end

        %% Real-time Request Flow
        Client -- HTTPS Request --> BFF
        BFF -- "GraphQL Query" --> MongoDB
        BFF -- "Internal API Call (Search/AI)" --> Agent
        Agent -- "Vector Search/Upsert" --> QdrantDB
        Agent -- "AI Task (e.g., Summarization)" --> LLM_API

        %% Offline Data Pipeline Flow
        Trigger --> StepFunction
        StepFunction -- "Start Fetch" --> LambdaFetch
        LambdaFetch -- "Read/Write Raw JSON" --> S3
        StepFunction -- "Start Processing" --> LambdaProcess
        LambdaProcess -- "Read Raw, Write Staged JSON" --> S3
        StepFunction -- "Start Final Assembly" --> LambdaWrite
        LambdaWrite -- "Read Staged JSON" --> S3
        LambdaWrite -- "Write Enriched Vectors" --> QdrantDB
    end

    style BFF fill:#f9f,stroke:#333,stroke-width:2px
    style Agent fill:#ccf,stroke:#333,stroke-width:2px
```
## Phase 6: Future Architecture - User Personalization & Recommendations

This phase outlines the architectural evolution required to support user accounts, watch history, and personalized recommendations. It builds upon the foundational BFF/Agent model established in Phase 5.

### Task 6.1: User Data Storage Strategy

- **Status:** `Future`
- **Goal:** Establish a scalable and efficient storage solution for user-generated data.
- **Decision:** We will extend our use of **MongoDB Atlas** to store all user-related data. While a relational database like PostgreSQL is the classic choice for user accounts due to its strong transactional integrity, using MongoDB offers a significant advantage by maintaining a single, unified database technology stack. This simplifies development, operations, and data access patterns for the BFF.
- **New Data Collections:**
  - **MongoDB `users` collection:** Will store user profile documents, including account information, credentials, and user settings.
  - **MongoDB `watch_history` collection:** Will store a granular log of user viewing activity. Each document will represent a single user's interaction with a single episode.
  - **Qdrant `users` collection:** A new vector collection will be created. Each point will represent a user, and its vector will be a mathematical summary of that user's tastes, derived from their watch history.
- **Scalability:** The `watch_history` collection is expected to grow to billions of entries. This scale will be managed efficiently through:
  - **Indexing:** A compound index on `(userId, updatedAt)` will ensure that queries for a specific user's history are instantaneous.
  - **Sharding:** As the collection grows, it will be sharded by `userId`, distributing the load across multiple servers and ensuring high performance.

### Task 6.2: Expanded Role of the BFF Service

- **Status:** `Future`
- **Goal:** Evolve the BFF from a data hydration layer into a full-fledged backend service that manages the lifecycle of user content.
- **New Responsibilities:**
  - **Authentication & Authorization:** The BFF will be responsible for handling user login and validating session tokens (e.g., JWTs) on all user-specific requests.
  - **CRUD Operations:** The BFF will expose new GraphQL mutations to handle Create, Read, Update, and Delete (CRUD) operations for user data. This will be the primary mechanism for the frontend to report user activity.
- **Example Write Workflow (Updating Watch History):**
  1. Frontend sends a GraphQL mutation: `updateWatchProgress(episodeId: "...", progress: 95)`.
  2. The BFF authenticates the user from the request.
  3. The mutation's resolver in the BFF directly calls the MongoDB client to `updateOne` document in the `watch_history` collection where the `userId` matches the authenticated user.

### Task 6.3: Recommendation Engine Architecture

- **Status:** `Future`
- **Goal:** Implement a personalized recommendation system by applying our existing vector search architecture to the user domain.
- **Core Concept: The "User Taste Vector"**
  - A new offline process will be created to generate a "taste vector" for each active user.
  - This process will read a user's `watch_history` from MongoDB, fetch the corresponding anime vectors from the `animes` collection in Qdrant, and compute a single, averaged vector representing the user's preferences.
  - This taste vector will be stored in the new `users` collection in Qdrant.
- **Recommendation Query Flow:**
  1. The BFF requests recommendations for an authenticated user.
  2. The BFF/Agent Service retrieves the user's "taste vector" from the **Qdrant `users` collection**.
  3. The Agent Service uses this vector to perform a similarity search against the **Qdrant `animes` collection**.
  4. The Agent Service returns a ranked list of `anime_id`s to the BFF.
  5. The BFF hydrates these IDs from the **MongoDB `animes` collection** and serves the results.

### Task 6.4: Data Model - User Watch History

- **Status:** `Future`
- **Goal:** Define the granular data model for storing user viewing events.
- **Example `watch_history` Document in MongoDB:**
  ```json
  {
    "_id": "unique_watch_event_id",
    "userId": "user_abc_uuid",
    "animeId": "one_piece_anime_uuid",
    "episodeId": "one_piece_ep1_uuid",
    "progress": 100,
    "status": "watched",
    "watchedAt": "2025-11-10T10:00:00Z",
    "updatedAt": "2025-11-10T10:00:00Z"
  }
  ```
- **Data Linking:** The explicit reference linking a user to an anime is stored in this MongoDB collection. There is no direct pointer between the user and anime collections in Qdrant; that link is made dynamically at query time via semantic similarity.

### Task 6.5: User Data & Compliance

The V1 implementation of this service only processes public, non-personal anime metadata. This section outlines the technical requirements that must be addressed if future features introduce Personally Identifiable Information (PII), such as user accounts (e.g., via MAL/AniDB logins) or personal watch lists.

- **Data Lifecycle & Erasure:** When features involving user data are added, a mechanism to permanently delete all data associated with a specific user upon their request (the "Right to Erasure") must be implemented. This process is distinct from the "orphaning" of general anime metadata.

- **Data Access & Portability:** The GraphQL API must be extended to allow an authenticated user to access and export all of their personal data in a machine-readable format.

- **Data Residency:** When user data is stored, the geographic region for all relevant data stores (e.g., MongoDB Atlas) must be explicitly chosen and documented to comply with data residency laws.

- **Consent Management:** A system for obtaining, recording, and managing user consent for data processing will be a prerequisite for any feature that collects PII.
