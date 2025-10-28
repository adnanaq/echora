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
      - For `AnimeEntry` objects with partially failed images, store the **new CloudFront URL** (e.g., `https://<cloudfront-domain>/images/<content_hash>.<extension>`) for successful images. For failed images, include the `image_url` and `error_message` within the `AnimeEntry` in DynamoDB as a placeholder.
      - This ensures we own and control all image assets from the start, with robust deduplication, retry logic, and comprehensive error handling. The CloudFront URLs stored in `AnimeEntry` will always be HTTPS-only.
  3.  **Data Source Clarity (`anime-offline-database.json` vs `enriched_anime_database.json`):**
      - **`anime-offline-database.json`:** This external file is the **raw, external source of truth** for basic anime metadata. It is never directly ingested into our production DynamoDB or Qdrant.
      - **`enriched_anime_database.json` (Initial Local Version):** This is a **derived artifact** created locally during initial data preparation. It contains `AnimeEntry` objects after programmatic enrichment and image re-hosting (with CloudFront URLs). This local file is manually uploaded to S3 during initial system bootstrap.
      - **`enriched_anime_database.json` (S3 Version):** Once uploaded to S3 (`s3://<bucket-name>/processed/enriched_anime_database.json`), this becomes the **initial snapshot of our system\'s enriched data**. It serves as the baseline for our weekly sync process and as a full, portable backup.
      - **DynamoDB (`anime-enriched-data` table):** This table is the **live, operational source of truth** for our enriched `AnimeEntry` objects after the initial bootstrap and subsequent weekly updates.
  4.  **Upload Enriched Database:** Manually upload the locally-generated `enriched_anime_database.json` (now with our S3 image URLs) to `s3://<bucket-name>/processed/enriched_anime_database.json`.
  5.  **Attach Metadata (Critical):** During the upload, attach a custom metadata tag `x-amz-meta-source-commit-sha` containing the commit hash of the `anime-offline-database` version that was used to generate the file. This provides the initial state for the system.
  6.  **Run Bulk Indexing Job:** Trigger a one-time, manual process to populate the Qdrant database. This will be a "scatter-gather" Lambda pattern to process ~40,000 entries in parallel without hitting timeouts.

### Task 1.0.1: Establish Infrastructure as Code (IaC) Framework

- **Status:** `To-Do`
- **Goal:** Define and implement a robust Infrastructure as Code (IaC) framework for provisioning and managing all cloud resources.
- **Rationale:** IaC ensures that our cloud infrastructure (S3, DynamoDB, EC2, Qdrant, Lambda, Step Functions, SQS, EventBridge, ElastiCache, API Gateway, CloudFront, Fargate) is provisioned, configured, and managed in a repeatable, auditable, and version-controlled manner. This minimizes manual errors, facilitates environment consistency, and supports efficient disaster recovery.
- **Decision:** We will use **Pulumi with Python** as our Infrastructure as Code (IaC) framework.
- **Rationale:** This choice aligns with our team's existing Python expertise, allowing us to use a single language for both application and infrastructure code. It enables powerful abstractions, simplifies testing, and reduces context-switching, leading to more seamless integration and faster development velocity.
- **Steps:**
  1.  **Tool Setup & Configuration:**
      - Install and configure the Pulumi CLI.
      - Set up the Pulumi project and stack configurations for our environments (dev, staging, prod).
      - Configure remote state management using the Pulumi Service backend to ensure state integrity and support collaborative development.
  2.  **Module Design:**
      - Design reusable IaC modules for common resource patterns (e.g., VPC, S3 buckets, DynamoDB tables, EC2 instances, Lambda functions).
      - Ensure modules are parameterized to support different environments (dev, staging, prod).
  3.  **Initial Resource Definition:**
      - Define all resources outlined in Phase 1 (S3 bucket, DynamoDB table, EC2 for Qdrant, etc.) using the chosen IaC framework.
      - Integrate the `docker-compose.yml` deployment for Qdrant on EC2 within the IaC (e.g., using `cloud-init` or a custom script for EC2 setup).
  4.  **CI/CD Integration:** Integrate IaC deployments into a CI/CD pipeline (e.g., GitHub Actions) to automate `plan` and `apply` operations, ensuring changes are reviewed and applied consistently.

### Task 1.1: Automated Weekly Database Sync

- **Status:** `To-Do`
- **Component:** `Weekly-Sync-Starter-Lambda`
- **Trigger:** AWS EventBridge, scheduled for every Saturday at 01:00 UTC.

- **Logic (Intelligent Sync):**
  1.  Uses the commit SHA in the S3 object metadata to check if the `anime-offline-database` has been updated.
  2.  If so, it downloads the new offline DB.
  3.  It iterates through each anime in the new offline DB and checks for its existence in the main `anime-enriched-data` table.
  4.  **For New Anime (not found in the main table):**
      - **Acquire Lock:** It attempts to acquire a lock by writing an item to the `anime-processing-locks` table using a conditional expression (`attribute_not_exists(anime_id)`).
      - **On Success:** If the lock is acquired successfully, it triggers the full `Enrichment-Step-Function` to add the new anime. The Step Function will be responsible for deleting the lock upon completion.
      - **On Failure:** If the lock acquisition fails (meaning another process has already claimed this anime), it does nothing and moves to the next anime.
  5.  **For Existing Anime (found in DynamoDB):**
      - If the entry is currently marked with `system_status == "ORPHANED"`, it will be "resurrected" by setting its status back to `ACTIVE` and clearing the `orphaned_at` timestamp.
      - Performs a selective "diff-and-merge" against the record fetched from DynamoDB to avoid data regression:
      - **Check `status`:** If the `status` in the offline file differs from our record, flag the anime for human review in the validation queue.
      - **Check `episodes` (count):** Compare the integer `episodes` count from the offline file with the actual number of episodes stored in our database.
        - If `offline_count > our_count`, flag the anime for human review, as we may be missing episodes.
        - If `offline_count <= our_count`, **do nothing**, as our live data is considered more accurate.
      - All other fields from the offline DB for existing entries are ignored to protect our enriched data.
  6.  **For Removed Anime (Orphaning):**
      - After processing the source file, the sync process will identify all anime present in our DynamoDB table that were not present in the source file.
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

### Task 1.2.1: Provision Enriched Data Store (DynamoDB)

- **Status:** `To-Do`
- **Goal:** Deploy a managed NoSQL database to store enriched `AnimeEntry` objects for fast retrieval and serving user-facing applications. This database will complement the vector database by providing full payload details for anime IDs returned by vector searches.
- **Rationale:** The existing JSON file approach for enriched data lacks the scalability, query flexibility, and operational features required for a production-grade, user-facing data store. DynamoDB offers schema flexibility, high performance, and seamless integration with the existing AWS ecosystem.
- **Steps:**
  1.  **Provision DynamoDB Table:** Create a new DynamoDB table (e.g., `anime-enriched-data`) in the appropriate AWS region.
      - **Primary Key:** Configure `anime_id` (string) as the Partition Key. **CRITICAL: This `anime_id` will be the common unique identifier used as the Point ID in Qdrant and the Primary Key in DynamoDB, ensuring a direct link between vector search results and detailed data.**
      - **Indexing:** A Global Secondary Index (GSI) must be provisioned to allow for efficient querying on the `system_status` field. This is critical for filtering out `ORPHANED` records from user-facing queries by default.
      - **Capacity Mode:** Start with On-Demand capacity for flexibility, or provisioned capacity if usage patterns are well-understood.
  2.  **Define Data Model:** The `AnimeEntry` Pydantic model (`src/models/anime.py`) will serve as the direct schema for documents stored in this table. Each `AnimeEntry` object will be stored as a single JSON document.
      - **Projection Expressions:** When retrieving data, utilize DynamoDB's Projection Expressions to fetch only the necessary attributes, avoiding overfetching for initial frontend renders (e.g., fetching only core metadata without full episode/character details).
  3.  **Configure IAM Roles/Policies:** Ensure that the FastAPI service (running on Fargate), Lambda functions (for ingestion/consolidation), and any other relevant services have appropriate IAM permissions (`dynamodb:GetItem`, `dynamodb:PutItem`, `dynamodb:UpdateItem`, `dynamodb:BatchGetItem`, etc.) to interact with the new DynamoDB table.
  4.  **Local Development Setup:** Integrate `amazon/dynamodb-local` into `docker-compose.yml` to provide a local, API-compatible DynamoDB instance for development and testing. Update `vector-service` environment variables (`DYNAMODB_ENDPOINT_URL`, `DYNAMODB_REGION`, `DYNAMODB_TABLE_NAME`) to point to the local instance during development.
  5.  **Initial Data Load Script:** Develop a one-time script to read the existing `enriched_anime_database.json` from S3 and batch-write all `AnimeEntry` objects into the new DynamoDB table. This will be part of the initial system bootstrap.

### Task 1.2.2: Provision Idempotency Lock Table

- **Status:** `To-Do`
- **Goal:** To create a dedicated DynamoDB table to act as a distributed lock, ensuring that enrichment workflows are only triggered once per new anime.
- **Rationale:** This prevents race conditions and duplicate workflow executions caused by Lambda retries, without polluting the primary `anime-enriched-data` table with placeholder records.
- **Steps:**
  1.  **Provision DynamoDB Table:** Create a new, simple DynamoDB table (e.g., `anime-processing-locks`).
      - **Primary Key:** Configure `anime_id` (string) as the Partition Key.
      - **TTL (Time-to-Live):** Enable TTL on a `ttl` attribute. This will automatically clean up stale locks from failed workflows after a set period (e.g., 24 hours).
  2.  **Configure IAM Roles/Policies:** Update the IAM role for the `Weekly-Sync-Starter-Lambda` to allow it to read, write, and delete items in this new table.

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
  2.  **Assemble Entry:** A state that combines the outputs of all previous steps into a single, final `AnimeEntry` object.
  3.  **Pause for Validation:** A single **pause state** that sends the complete `AnimeEntry` object to the validation queue and waits for a human to approve, edit, or reject it.
  4.  **Commit Data:** Upon approval, this state runs. It performs a partial update to Qdrant and then commits the data to our primary stores:
      - It **writes the approved `AnimeEntry` object to the DynamoDB table**, making it live in the system.
      - It saves a copy of the single, approved `AnimeEntry` object to a dedicated S3 prefix (e.g., `processed/updated-entries/<anime-id>.json`) as a permanent audit log.
  5.  **Cleanup:** A final state to clean up any temporary resources from EFS.

### Task 1.4: Implement Portability & Backup Job

- **Status:** `To-Do`
- **Component:** `create-database-snapshot-lambda`
- **Trigger:** EventBridge (scheduled, e.g., weekly on Sunday).
- **Goal:** To create a periodic, full, cloud-agnostic snapshot of the entire enriched database. This file serves as a crucial artifact for disaster recovery and simplifies potential future migrations to other cloud providers.
- **Logic:**
  1.  The Lambda is triggered by a weekly schedule.
  2.  It performs a full scan of the `anime-enriched-data` DynamoDB table to fetch every record.
  3.  It assembles all the records into a single list.
  4.  It writes the complete list to the `enriched_anime_database.json` file in S3, overwriting the previous week's snapshot. This file now represents a complete, portable backup of the database state.

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
        - `dynamodb-local`: The `amazon/dynamodb-local` image.
        - `minio`: To act as an S3-compatible object store.
        - `signoz`: The full SigNoz observability stack.
  2.  **Configuration Management:**
      - Create a `.env.local.example` file containing all necessary environment variables for the `vector-service` to connect to the other local containers (e.g., `DYNAMODB_ENDPOINT_URL=http://dynamodb-local:8000`).
      - This file will be git-ignored and developers can copy it to `.env.local` for their specific setup.
  3.  **Data Seeding:**
      - Create a `scripts/seed_local_env.py` script.
      - This script will populate the local Qdrant and DynamoDB instances with a small, consistent set of sample data, allowing developers to have a working environment immediately after setup.
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
  1.  **Handle Ongoing Shows:** Queries the `anime-enriched-data` DynamoDB table for all entries with `status == "ONGOING"`. For each, it finds the next episode scheduled to air in the next 24 hours and creates a one-time EventBridge schedule to trigger the `run-single-episode-update-lambda` at the target time.
      - **Idempotency:** To prevent duplicate schedules upon retry, a deterministic, unique name will be used for each schedule (e.g., `animeId-{anime_id}-episode-{episode_number}`). An attempt to create a schedule with a name that already exists will be caught and treated as a success.
  2.  **Handle Upcoming Shows:** Queries the DynamoDB table for all entries with `status == "UPCOMING"` and a premiere date within the next 24 hours. For each of these anime, it proactively triggers the full `Enrichment-Step-Function`.
      - **Idempotency:** This process will use the same locking mechanism described in Task 1.1 (using the `anime-processing-locks` table) to prevent duplicate workflow triggers for the same premiere.

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

## Phase 4: Caching & Performance Optimization

### Task 4.1: Provision Caching Infrastructure

- **Status:** `To-Do`
- **Component:** Amazon ElastiCache for Redis
- **Goal:** To deploy a managed Redis cluster within the VPC to serve as a high-speed cache.
- **Configuration:** A small cache instance (e.g., `cache.t3.small`) will be provisioned in a private subnet, with a security group that only allows access from the Lambda functions and the Fargate service.

### Task 4.2: Integrate Caching for External API Calls

- **Status:** `To-Do`
- **Component:** API helper classes (e.g., `AnilistHelper`, `JikanHelper`)
- **Goal:** To reduce redundant API calls, avoid rate-limiting, and speed up the enrichment process.
- **Logic:**
  1.  The core data fetching method in each helper will be modified.
  2.  Before making a live HTTP request, it will first check the Redis cache for the requested data using a standardized key (e.g., `jikan:anime:123`).
  3.  **On a cache hit,** it will return the cached data immediately.
  4.  **On a cache miss,** it will perform the real API request, save the result to the Redis cache with an appropriate TTL (Time-To-Live), and then return the data.

### Task 4.2.1: Integrate Caching for Enriched Data (DynamoDB)

- **Status:** `To-Do`
- **Component:** FastAPI Service, Lambda functions (Enrichment Step Function)
- **Goal:** To significantly reduce latency and DynamoDB read costs for retrieving `AnimeEntry` objects.
- **Logic:**
  1.  **Caching Mechanism:** Utilize Amazon ElastiCache for Redis (provisioned in Task 4.1) as a distributed, in-memory cache.
  2.  **Caching Strategy (Cache-Aside):**
      - **Read Path:** When the FastAPI service needs to retrieve an `AnimeEntry` from DynamoDB (e.g., after a vector search), it will first check the Redis cache using a standardized key (`anime:<anime_id>`).
      - **Cache Hit:** If the `AnimeEntry` is found in Redis, it will be returned immediately.
      - **Cache Miss:** If not found, the `AnimeEntry` will be fetched from DynamoDB (using Projection Expressions to optimize retrieval), stored in Redis with a TTL, and then returned.
  3.  **Cache Invalidation:**
      - When an `AnimeEntry` is **created or updated** in DynamoDB by the `Enrichment Step Function` (Task 1.3), the corresponding cache entry in Redis will be explicitly **deleted**. This ensures data consistency by forcing subsequent reads to fetch the fresh data from DynamoDB.
  4.  **Cache Key:** `anime:<anime_id>`
  5.  **Cache Value:** JSON representation of the `AnimeEntry` object.
  6.  **Cache Validity (TTL):** `24 hours (86400 seconds)`. The explicit invalidation mechanism ensures consistency, while the TTL acts as a safety net.

### Task 4.3: Enable API Gateway Caching

- **Status:** `To-Do`
- **Component:** Amazon API Gateway
- **Goal:** To reduce latency and cost for frequent, identical user-facing search queries.
- **Logic:**
  1.  Enable the built-in caching feature on the API Gateway stage.
  2.  Configure the cache to use the request path and query parameters as the cache key.
  3.  Set a reasonable TTL (e.g., 5-15 minutes) to serve repeated public queries directly from the cache, bypassing our Fargate service entirely.

## Phase 5: API Strategy

This phase outlines the design of the user-facing API, which is composed of four distinct endpoint types, each with a tailored caching strategy to ensure performance and efficiency. The API will be a FastAPI application running on AWS Fargate, fronted by Amazon API Gateway.

### Caching Layers Overview

The API will leverage two independent layers of caching:

1.  **API Gateway Cache:** An "edge" cache that stores full HTTP responses for specific requests. It is keyed by the request path and query parameters. When a hit occurs, the request does not reach our backend service.
2.  **Redis (ElastiCache) Cache:** An application-level cache that stores processed data, primarily `AnimeEntry` objects fetched from DynamoDB. It is keyed by `anime_id`.

### Endpoint Type 1: Agentic Search

- **Description:** A natural language search endpoint (e.g., `/search/agentic`). The user's raw query is first processed by an "Agentic AI" layer. This layer will be implemented by calling a Qwen3 model hosted on **Amazon Bedrock**. The model's role is to parse the user's intent and generate a refined query and a structured filter object, which is then used to query Qdrant.
- **Caching Strategy:**
  - **API Gateway Cache:** Less effective due to the high variability of natural language queries. May be disabled for this endpoint.
  - **Redis Cache:** Very effective. Caches the `AnimeEntry` objects for the results returned by Qdrant. This saves DynamoDB reads when different searches return overlapping results.

### Endpoint Type 2: Structured Search

- **Description:** A filtered search endpoint (e.g., `/search/structured`) that accepts specific query parameters from the frontend UI (`?genre=Action&year=2023`). It queries Qdrant directly.
- **Caching Strategy:**
  - **API Gateway Cache:** Extremely effective. Caches full responses for common filter combinations, providing instant results and reducing backend load significantly.
  - **Redis Cache:** Acts as a second layer, caching individual `AnimeEntry` objects if the API Gateway has a cache miss.

### Endpoint Type 3: Third-Party Proxy

- **Description:** Endpoints that act as a facade for external APIs (e.g., MAL, AniList). This centralizes all external communication through our backend.
- **Caching Strategy:**
  - **API Gateway Cache:** Primary and highly effective. Caches the responses from the third-party APIs for a configured TTL (e.g., 1-24 hours). This is critical for avoiding rate-limiting by external services.
  - **Redis Cache:** Generally not used for this endpoint type.

### Endpoint Type 4: Direct Database Queries

- **Description:** Endpoints that query DynamoDB directly for specific data sets, without involving Qdrant.
- **Use Cases & Caching Strategy:**
  - **Get Single Entry (`/anime/{id}`):**
    - **API Gateway & Redis Caches:** Both are extremely effective. A request must miss both caches to hit DynamoDB.
  - **Get Curated Lists (`/popular`, `/top-rated`):**
    - **API Gateway Cache:** Extremely effective for caching the final generated list for a set TTL.
    - **Redis Cache:** Can be used to store a pre-computed list. A background job can run periodically (e.g., once every 12 hours) to query the database and refresh this list in Redis, making the API endpoint a simple and fast read operation.
