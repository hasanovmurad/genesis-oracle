# Protocol Architecture Design – Spreeland Logistics

## 1. Infrastructure Discovery: MCP

The Dispatch Agent uses the Model Context Protocol to access the city's bridge-status infrastructure.

A PostgreSQL-aware MCP server exposes controlled tools such as:

- `get_bridge_status`
- `list_open_routes`
- `get_river_level`

The dispatcher does not receive unrestricted database access. Instead, it discovers and invokes the tools published by the MCP server. This separates the agent from database credentials and implementation details.

## 2. Expert Consultation: A2A

The dispatcher discovers the external Weather-Predictor Agent through its A2A Agent Card.

The Agent Card describes:

- agent identity,
- service endpoint,
- supported capabilities,
- available skills,
- streaming support,
- accepted input and output formats.

After checking the advertised weather-forecasting skill, the dispatcher sends a task to the remote agent and receives the predicted river and storm conditions.

## 3. Secure Fulfillment: UCP and AP2

UCP manages the commercial process for purchasing two tons of gherkins:

1. Supplier discovery
2. Catalog query
3. Offer comparison
4. Cart creation
5. Checkout preparation

AP2 provides the payment authorization layer. The owner creates a cryptographically signed mandate containing:

- maximum authorized amount,
- allowed merchant,
- product category,
- quantity,
- validity period.

The payment may proceed only when the proposed transaction satisfies the mandate. The signed transaction record provides an auditable trail.

## 4. Dynamic Visualization: A2UI and AG-UI

A2UI describes the delivery dashboard as a declarative JSON structure instead of custom frontend code.

The schema contains components such as:

- delivery status,
- selected route,
- bridge condition,
- weather warning,
- purchase authorization,
- estimated arrival time.

AG-UI transports events between the dispatcher backend and the user interface. It streams route updates, tool calls, warnings, and agent responses while also accepting user confirmations.

## Protocol Flow

1. MCP retrieves bridge and river data.
2. A2A delegates weather prediction to a specialist agent.
3. UCP discovers suppliers and prepares the order.
4. AP2 verifies the owner's payment mandate.
5. A2UI defines the delivery dashboard.
6. AG-UI streams live logistics updates.