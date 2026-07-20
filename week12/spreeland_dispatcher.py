import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator


# ---------------------------------------------------------
# MCP BENZETİMİ: Altyapı araçları
# ---------------------------------------------------------

def get_bridge_status(bridge_id: str) -> dict:
    """
    Returns the current status of a bridge.

    Args:
        bridge_id: Identifier of the bridge.

    Returns:
        Bridge status data.
    """
    bridge_data = {
        "B97": {
            "status": "OPEN",
            "load_limit_tons": 12.0,
            "maintenance": False,
        },
        "L49": {
            "status": "CLOSED",
            "load_limit_tons": 0.0,
            "maintenance": True,
        },
        "SPREE-01": {
            "status": "RESTRICTED",
            "load_limit_tons": 3.0,
            "maintenance": False,
        },
    }

    return bridge_data.get(
        bridge_id,
        {
            "status": "UNKNOWN",
            "load_limit_tons": 0.0,
            "maintenance": False,
        },
    )


def get_river_level() -> dict:
    """
    Returns the current river level in meters.
    """
    return {
        "river": "Spree",
        "level_m": 1.82,
        "risk": "NORMAL",
    }


# ---------------------------------------------------------
# A2A: Agent Card ve hava tahmin ajanı
# ---------------------------------------------------------

WEATHER_AGENT_CARD = {
    "name": "weather-predictor-agent",
    "description": "Specialized agent for regional weather and flood prediction.",
    "endpoint": "https://weather-agent.example/a2a",
    "skills": [
        {
            "id": "spreewald-weather-forecast",
            "name": "Spreewald Weather Forecast",
            "description": "Predicts rainfall, storms, and river-level risks.",
        }
    ],
    "streaming": True,
    "input_modes": ["application/json"],
    "output_modes": ["application/json"],
}


def discover_weather_agent(required_skill: str) -> dict | None:
    """
    Checks whether the external A2A agent provides the requested skill.
    """
    for skill in WEATHER_AGENT_CARD["skills"]:
        if skill["id"] == required_skill:
            return WEATHER_AGENT_CARD

    return None


def request_weather_forecast(agent_card: dict) -> dict:
    """
    Simulates an A2A request to a remote weather agent.
    """
    return {
        "agent": agent_card["name"],
        "forecast": {
            "rain_probability_percent": 35,
            "storm_warning": False,
            "expected_river_change_m": 0.08,
        },
        "recommendation": "Normal routing conditions expected.",
    }


# ---------------------------------------------------------
# UCP + AP2: Satın alma ve ödeme yetkisi
# ---------------------------------------------------------

@dataclass
class PurchaseMandate:
    owner: str
    merchant: str
    product: str
    quantity_tons: float
    maximum_amount_eur: float
    approved: bool


def prepare_ucp_order() -> dict:
    """
    Simulates a UCP commerce flow for purchasing gherkins.
    """
    return {
        "merchant": "Spreewald Organic Cooperative",
        "product": "Organic gherkins",
        "quantity_tons": 2.0,
        "unit_price_eur_per_ton": 1400.0,
        "total_amount_eur": 2800.0,
        "checkout_status": "PREPARED",
    }


def verify_ap2_mandate(order: dict, mandate: PurchaseMandate) -> dict:
    """
    Verifies whether the order satisfies the AP2 payment mandate.
    """
    valid = (
        mandate.approved
        and mandate.merchant == order["merchant"]
        and mandate.product == order["product"]
        and mandate.quantity_tons >= order["quantity_tons"]
        and mandate.maximum_amount_eur >= order["total_amount_eur"]
    )

    return {
        "authorized": valid,
        "reason": (
            "Transaction satisfies the signed owner mandate."
            if valid
            else "Transaction violates or lacks owner authorization."
        ),
    }


# ---------------------------------------------------------
# AG-UI: Olay akışı
# ---------------------------------------------------------

async def stream_event(event_type: str, payload: dict) -> AsyncIterator[str]:
    """
    Simulates an AG-UI event stream.
    """
    event = {
        "type": event_type,
        "payload": payload,
    }

    await asyncio.sleep(0.2)
    yield json.dumps(event)


# ---------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------

async def run_dispatcher() -> None:
    print("Starting Spreeland Dispatcher...\n")

    river = get_river_level()

    async for event in stream_event("infrastructure.river", river):
        print(event)

    bridges = {
        bridge_id: get_bridge_status(bridge_id)
        for bridge_id in ["B97", "L49", "SPREE-01"]
    }

    async for event in stream_event("infrastructure.bridges", bridges):
        print(event)

    available_routes = [
        bridge_id
        for bridge_id, status in bridges.items()
        if status["status"] in {"OPEN", "RESTRICTED"}
        and status["load_limit_tons"] >= 2.0
    ]

    weather_agent = discover_weather_agent(
        "spreewald-weather-forecast"
    )

    if weather_agent is None:
        raise RuntimeError("No compatible weather agent discovered.")

    forecast = request_weather_forecast(weather_agent)

    async for event in stream_event("a2a.weather_forecast", forecast):
        print(event)

    order = prepare_ucp_order()

    mandate = PurchaseMandate(
        owner="Spreeland Logistics GmbH",
        merchant="Spreewald Organic Cooperative",
        product="Organic gherkins",
        quantity_tons=2.0,
        maximum_amount_eur=3000.0,
        approved=True,
    )

    payment_result = verify_ap2_mandate(order, mandate)

    async for event in stream_event(
        "commerce.purchase_authorization",
        {
            "order": order,
            "ap2_result": payment_result,
        },
    ):
        print(event)

    selected_route = available_routes[0] if available_routes else None

    final_status = {
        "cargo": "2 tons of organic gherkins",
        "destination": "Cottbus",
        "selected_bridge": selected_route,
        "weather_risk": forecast["forecast"]["storm_warning"],
        "payment_authorized": payment_result["authorized"],
        "delivery_status": (
            "READY_FOR_DISPATCH"
            if selected_route and payment_result["authorized"]
            else "BLOCKED"
        ),
    }

    async for event in stream_event(
        "delivery.final_status",
        final_status,
    ):
        print(event)


if __name__ == "__main__":
    asyncio.run(run_dispatcher())