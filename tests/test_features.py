"""Training-serving parity: replaying transactions through the ONLINE path
(feature store + daily reference reload) must reproduce the OFFLINE training features."""
import fakeredis
import numpy as np
import pytest

from fraud.config import DAY_S, FEATURE_NAMES
from fraud.data import generate
from fraud.feature_store import InMemoryFeatureStore, RedisFeatureStore
from fraud.features import build_reference_tables, build_training_features, compute_features


@pytest.fixture(scope="module")
def parity_df():
    return generate(n_cards=120, days=30, fraud_events=40, seed=11)


@pytest.mark.parametrize("store_kind", ["memory", "redis"])
def test_online_features_match_training(parity_df, store_kind):
    offline = build_training_features(parity_df).set_index("txn_id")
    store = InMemoryFeatureStore() if store_kind == "memory" else RedisFeatureStore(fakeredis.FakeRedis())

    current_day, mismatches = None, 0
    for r in parity_df.sort_values(["ts", "txn_id"]).itertuples(index=False):
        day = int(r.ts) // DAY_S
        if day != current_day:  # the daily batch job refreshes reference tables
            store.load_reference(build_reference_tables(parity_df, day))
            current_day = day
        ref = store.get_reference()
        online = compute_features(ts=int(r.ts), amount=r.amount, location=r.location,
                                  history=store.get_history(r.card_id, int(r.ts)),
                                  loc_rate=ref.loc(r.location), mcc_rate=ref.mcc(r.merchant_category))
        expected = offline.loc[r.txn_id, FEATURE_NAMES].to_numpy(float)
        got = np.array([online[f] for f in FEATURE_NAMES])
        mismatches += int(not np.allclose(expected, got, rtol=1e-9, atol=1e-9))
        store.record_txn(r.card_id, r.txn_id, int(r.ts), r.amount, r.location)
    assert mismatches == 0


def test_no_self_leakage_and_idempotent_writes():
    store = RedisFeatureStore(fakeredis.FakeRedis())
    store.record_txn("c1", "t1", 1000, 10.0, "city_01")
    store.record_txn("c1", "t1", 1000, 10.0, "city_01")  # Kafka redelivery
    assert len(store.get_history("c1", 2000)) == 1       # counted once
    assert store.get_history("c1", 1000) == []           # a txn never sees itself


def test_history_window_is_30_days():
    store = RedisFeatureStore(fakeredis.FakeRedis())
    store.record_txn("c1", "old", 0, 5.0, "city_01")
    store.record_txn("c1", "new", 31 * DAY_S, 5.0, "city_01")
    assert [h.ts for h in store.get_history("c1", 31 * DAY_S + 1)] == [31 * DAY_S]
