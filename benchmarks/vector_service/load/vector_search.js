// k6 load test for VectorSearchService/Search.
//
// Every test type starts searches on a schedule (arrival rate), whether or not
// earlier searches have answered, so a slow service cannot hide its own
// slowdown by holding back the load.
//
// Settings (environment variables, passed with -e NAME=value):
//   TEST_TYPE    smoke | load | stress | spike | soak | breakpoint (default smoke)
//   TARGET       host:port of the vector service (default localhost:8001)
//   TLS          true connects with TLS, as a cloud endpoint needs (default false)
//   RATE         searches/s one instance should sustain (default 20)
//   STRESS_RATE  default RATE x 1.5
//   SPIKE_RATE   default RATE x 3
//   MAX_RATE     breakpoint ceiling (default RATE x 10)
//   HOLD         plateau length for load/stress (default 10m); soak uses SOAK_HOLD (default 3h)
//   BREAKPOINT_DURATION  ramp length for breakpoint (default 20m)
//   P95_MS       fail the run when p95 exceeds this (default: report only)
//   ABORT_P95_MS breakpoint stops once p95 passes this (default 2000)
//   QUERY_MIX    short,title,long weights (default 40,40,20)
//   UNIQUE       true appends a counter to every query, so no cache can answer it
//   LIMIT        results per search (default 10)
//   WITH_PAYLOAD false asks for IDs and scores only (default: payloads included)
//   TIMEOUT      per-search deadline (default 30s)
//   MAX_VUS      most searches in flight at once (default 2000)

import exec from 'k6/execution';
import grpc from 'k6/net/grpc';
import { check } from 'k6';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';

const TEST_TYPE = __ENV.TEST_TYPE || 'smoke';
const TARGET = __ENV.TARGET || 'localhost:8001';
const TLS = __ENV.TLS === 'true';
const RATE = Number(__ENV.RATE || 20);
const STRESS_RATE = Number(__ENV.STRESS_RATE || Math.ceil(RATE * 1.5));
const SPIKE_RATE = Number(__ENV.SPIKE_RATE || RATE * 3);
const MAX_RATE = Number(__ENV.MAX_RATE || RATE * 10);
const HOLD = __ENV.HOLD || '10m';
const SOAK_HOLD = __ENV.SOAK_HOLD || '3h';
const BREAKPOINT_DURATION = __ENV.BREAKPOINT_DURATION || '20m';
const ABORT_P95_MS = Number(__ENV.ABORT_P95_MS || 2000);
const UNIQUE = __ENV.UNIQUE === 'true';
const LIMIT = Number(__ENV.LIMIT || 10);
const WITH_PAYLOAD = __ENV.WITH_PAYLOAD !== 'false';
const TIMEOUT = __ENV.TIMEOUT || '30s';
const MAX_VUS = Number(__ENV.MAX_VUS || 2000);
const QUERY_KINDS = ['short', 'title', 'long'];
const QUERY_WEIGHTS = (__ENV.QUERY_MIX || '40,40,20').split(',').map(Number);

const queriesByKind = Object.fromEntries(
  QUERY_KINDS.map((kind) => [
    kind,
    new SharedArray(`queries_${kind}`, () => JSON.parse(open('./search_queries.json'))[kind]),
  ]),
);

const searchErrors = new Counter('search_errors');

const client = new grpc.Client();
client.load(['../../../protos'], 'vector_service/v1/vector_search.proto');
let connected = false;

function arrivalRate(stages) {
  return {
    executor: 'ramping-arrival-rate',
    startRate: 1,
    timeUnit: '1s',
    preAllocatedVUs: Math.min(MAX_VUS, 50),
    maxVUs: MAX_VUS,
    stages,
  };
}

const SCENARIOS = {
  smoke: {
    executor: 'constant-arrival-rate',
    rate: 1,
    timeUnit: '1s',
    duration: '30s',
    preAllocatedVUs: 5,
    maxVUs: 20,
  },
  load: arrivalRate([
    { duration: '2m', target: RATE },
    { duration: HOLD, target: RATE },
    { duration: '1m', target: 0 },
  ]),
  stress: arrivalRate([
    { duration: '5m', target: STRESS_RATE },
    { duration: HOLD, target: STRESS_RATE },
    { duration: '2m', target: 0 },
  ]),
  spike: arrivalRate([
    { duration: '30s', target: SPIKE_RATE },
    { duration: '1m', target: SPIKE_RATE },
    { duration: '30s', target: 0 },
    { duration: '2m', target: 0 },
  ]),
  soak: arrivalRate([
    { duration: '5m', target: RATE },
    { duration: SOAK_HOLD, target: RATE },
    { duration: '5m', target: 0 },
  ]),
  breakpoint: arrivalRate([{ duration: BREAKPOINT_DURATION, target: MAX_RATE }]),
};

if (!(TEST_TYPE in SCENARIOS)) {
  throw new Error(`Unknown TEST_TYPE "${TEST_TYPE}"; use one of ${Object.keys(SCENARIOS).join(', ')}`);
}

const thresholds = {
  checks: ['rate>0.99'],
  search_errors: ['count<1'],
};
// Listing a tagged sub-metric makes the summary report it on its own line.
for (const kind of QUERY_KINDS) {
  thresholds[`grpc_req_duration{query_kind:${kind}}`] = [];
}
if (__ENV.P95_MS) {
  thresholds.grpc_req_duration = [`p(95)<${Number(__ENV.P95_MS)}`];
}
if (TEST_TYPE === 'breakpoint') {
  thresholds.grpc_req_duration = [
    { threshold: `p(95)<${ABORT_P95_MS}`, abortOnFail: true, delayAbortEval: '30s' },
  ];
  thresholds.checks = [{ threshold: 'rate>0.95', abortOnFail: true, delayAbortEval: '30s' }];
}

export const options = {
  scenarios: { [TEST_TYPE]: SCENARIOS[TEST_TYPE] },
  thresholds,
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  tags: { test_type: TEST_TYPE },
};

function pickQueryKind() {
  const total = QUERY_WEIGHTS.reduce((sum, weight) => sum + weight, 0);
  let draw = Math.random() * total;
  for (let index = 0; index < QUERY_KINDS.length; index += 1) {
    draw -= QUERY_WEIGHTS[index];
    if (draw < 0) {
      return QUERY_KINDS[index];
    }
  }
  return QUERY_KINDS[0];
}

export default function () {
  if (!connected) {
    client.connect(TARGET, { plaintext: !TLS, timeout: TIMEOUT });
    connected = true;
  }
  const queryKind = pickQueryKind();
  const queries = queriesByKind[queryKind];
  let queryText = queries[Math.floor(Math.random() * queries.length)];
  if (UNIQUE) {
    queryText = `${queryText} ${exec.scenario.iterationInTest}`;
  }

  const response = client.invoke(
    'vector_service.v1.VectorSearchService/Search',
    { query_text: queryText, limit: LIMIT, with_payload: WITH_PAYLOAD },
    { timeout: TIMEOUT, tags: { query_kind: queryKind } },
  );

  const answered = check(
    response,
    {
      'status is OK': (reply) => reply && reply.status === grpc.StatusOK,
      'no error in response': (reply) => reply && reply.message && !reply.message.error,
    },
    { query_kind: queryKind },
  );
  if (!answered) {
    searchErrors.add(1, { query_kind: queryKind });
  }
}
