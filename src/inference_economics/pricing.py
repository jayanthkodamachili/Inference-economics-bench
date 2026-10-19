"""Where the price of a machine hour comes from.

Two paths, and the difference matters for anyone reading the published numbers.

The committed snapshot is what every figure in this repository uses. It is a static table
with the date it was taken, so the numbers can be reproduced exactly and so a reader can
see how stale they are.

The live path queries the AWS Price List API through boto3. It exists because a cost model
pinned to a snapshot silently rots, and because the question "what does this cost now" is
the one an operator actually has. It is not used for published figures, it is covered by
tests that stub the client rather than call AWS, and it is wired so that a failure to reach
AWS falls back to the snapshot rather than crashing a benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Snapshot of on demand Linux pricing, US East (N. Virginia), taken 2026-02-19.
# Committed deliberately: published costs must be reproducible, and a figure that changes
# because a vendor changed a price while nothing in this repository changed is a figure
# nobody can check. The date is part of the data for the same reason.
SNAPSHOT_DATE = "2026-02-19"
SNAPSHOT_REGION = "us-east-1"
PRICE_SNAPSHOT_USD_PER_HOUR = {
    "c7g.large": 0.0723,
    "c7i.large": 0.08925,
    "m7i.large": 0.1008,
    "r7i.large": 0.1323,
}
DEFAULT_INSTANCE = "c7i.large"
INSTANCE_VCPUS = {"c7g.large": 2, "c7i.large": 2, "m7i.large": 2, "r7i.large": 2}


class PricingError(RuntimeError):
    """Raised when a price cannot be established from any source."""


@dataclass(frozen=True)
class Price:
    """One instance hour, and where the number came from."""

    instance_type: str
    usd_per_hour: float
    source: str
    as_of: str

    def as_dict(self) -> dict:
        return {"instance_type": self.instance_type,
                "usd_per_hour": self.usd_per_hour,
                "source": self.source, "as_of": self.as_of}


def from_snapshot(instance_type: str = DEFAULT_INSTANCE) -> Price:
    if instance_type not in PRICE_SNAPSHOT_USD_PER_HOUR:
        raise PricingError(
            f"no snapshot price for {instance_type!r}; known types are "
            f"{sorted(PRICE_SNAPSHOT_USD_PER_HOUR)}"
        )
    return Price(instance_type, PRICE_SNAPSHOT_USD_PER_HOUR[instance_type],
                 source="snapshot", as_of=SNAPSHOT_DATE)


def _extract_on_demand_usd(price_item: str) -> float:
    """Pull the hourly USD figure out of one Price List API record.

    The shape is awkward and worth naming: terms.OnDemand is a dict keyed by an offer
    code, each with priceDimensions keyed by a rate code, each with pricePerUnit.USD as a
    string. Nothing in it is a list and none of the keys are stable, so it is walked rather
    than indexed.
    """
    document = json.loads(price_item)
    on_demand = document.get("terms", {}).get("OnDemand", {})
    for offer in on_demand.values():
        for dimension in offer.get("priceDimensions", {}).values():
            usd = dimension.get("pricePerUnit", {}).get("USD")
            if usd is not None and float(usd) > 0:
                return float(usd)
    raise PricingError("no non zero on demand USD price in the response")


def from_aws(instance_type: str = DEFAULT_INSTANCE, region: str = SNAPSHOT_REGION,
             client=None, fall_back: bool = True) -> Price:
    """Query the AWS Price List API, falling back to the snapshot.

    The fallback is deliberate. A benchmark that cannot run because a pricing endpoint is
    unreachable has made a reporting detail into a hard dependency, and the snapshot is
    right there. What it must not do is pretend: the returned Price records which source
    was used, and every published table prints it.
    """
    try:
        if client is None:
            import boto3

            # The Price List API is only served from a few regions regardless of which
            # region is being priced, which is why this is not the region argument.
            client = boto3.client("pricing", region_name="us-east-1")

        response = client.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            ],
            MaxResults=1,
        )
        items = response.get("PriceList", [])
        if not items:
            raise PricingError(f"AWS returned no price for {instance_type} in {region}")
        return Price(instance_type, _extract_on_demand_usd(items[0]),
                     source="aws-price-list", as_of="live")
    except Exception as exc:  # noqa: BLE001
        if not fall_back:
            raise PricingError(f"could not price {instance_type}: {exc}") from exc
        return from_snapshot(instance_type)
