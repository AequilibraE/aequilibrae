# VISUM GeoJSON Field Inventory

No production VISUM sample files are currently checked into this repository. The inventory below is therefore based on
the compact VISUM-like fixture contract used by this change and the field semantics captured in `design.md` and
`reviewers.md`. If real agency VISUM exports are added later, this document should be refreshed from those files before
changing importer defaults.

## Layer Inventory

### `node`

| Field | Type | Nulls | Example values | Role |
| --- | --- | --- | --- | --- |
| `NO` | integer | 0 | `1`, `2` | Required source node ID |
| `NAME` | text | optional | `Main`, `West` | Preserved-only metadata |
| `geometry` | point | 0 | `POINT (-46.38 -23.55)` | Required node geometry |

### `link`

| Field | Type | Nulls | Example values | Role |
| --- | --- | --- | --- | --- |
| `NO` | integer | 0 | `100` | Required source link ID |
| `FROMNODENO` | integer | 0 | `1` | Required AB endpoint reference |
| `TONODENO` | integer | 0 | `2` | Required AB endpoint reference |
| `TSYSSET` | text/list | directional | `CAR,HGV`, `CAR`, empty | Required AB transport systems when AB is available |
| `R_TSYSSET` | text/list | directional | `CAR`, empty | Required BA transport systems when BA is available |
| `TYPENO` | integer/text | optional | `1`, `10` | AB link-type mapping candidate |
| `R_TYPENO` | integer/text | optional | `1`, `20` | BA link-type mapping candidate or diagnostic metadata |
| `LC` | text/integer | optional | `ARTERIAL` | AB link class; primary default link-type mapping field |
| `R_LC` | text/integer | optional | `LOCAL` | BA link class; diagnostic metadata unless split in a later version |
| `LENGTH` | unit string/number | optional | `0.548km`, `548m` | AB source length |
| `R_LENGTH` | unit string/number | optional | `0.550km` | BA source length |
| `V0PRT` | unit string/number | optional | `50km/h` | AB free-flow speed |
| `R_V0PRT` | unit string/number | optional | `45km/h` | BA free-flow speed |
| `CAPPRT` | unit string/number | optional | `1200veh/h` | AB assignment capacity |
| `R_CAPPRT` | unit string/number | optional | `1100veh/h` | BA assignment capacity |
| `T0PRT` | unit string/number | optional | `0.75min` | AB free-flow travel time override |
| `R_T0PRT` | unit string/number | optional | `0.8min` | BA free-flow travel time override |
| `geometry` | linestring | 0 | `LINESTRING (...)` | Required link geometry |

### `zone_centroid`

| Field | Type | Nulls | Example values | Role |
| --- | --- | --- | --- | --- |
| `NO` | integer | 0 | `1001` | Required source zone/centroid ID |
| `NAME` | text | optional | `CBD` | Zone name |
| `geometry` | point | 0 | `POINT (...)` | Required centroid geometry |

### `zone_polygon`

| Field | Type | Nulls | Example values | Role |
| --- | --- | --- | --- | --- |
| `NO` | integer | 0 | `1001` | Optional zone ID matching `zone_centroid.NO` |
| `NAME` | text | optional | `CBD` | Zone name |
| `geometry` | polygon/multipolygon | optional | `POLYGON (...)` | Optional context zone geometry |

### `connector`

| Field | Type | Nulls | Example values | Role |
| --- | --- | --- | --- | --- |
| `NO` | integer | 0 | `9001` | Required source connector ID |
| `ZONENO` | integer | 0 | `1001` | Required zone centroid reference |
| `NODENO` | integer | 0 | `1` | Required network node reference |
| `TSYSSET` | text/list | directional | `CAR,HGV` | Connector private modes |
| `R_TSYSSET` | text/list | optional | `CAR,HGV`, empty | Reverse connector modes |
| `LENGTH` | unit string/number | optional | `120m` | Source connector length |
| `V0PRT` | unit string/number | optional | `30km/h` | Source connector speed |
| `CAPPRT` | unit string/number | optional | `9999veh/h` | Source connector capacity |
| `geometry` | linestring | 0 | `LINESTRING (...)` | Required centroid-to-node geometry |

### `countlocation`

| Field | Type | Nulls | Example values | Role |
| --- | --- | --- | --- | --- |
| `NO` | integer | 0 | `5001` | Count-location source ID |
| `LINKNO` | integer | optional | `100` | Supported link-count association |
| `FROMNODENO` | integer | optional | `1` | Direction/reference validation |
| `TONODENO` | integer | optional | `2` | Direction/reference validation |
| `CAR_ORIG` | numeric | optional | `950` | Observed car count candidate |
| `HVG_ORIG` | numeric | optional | `120` | Observed heavy-vehicle count candidate |
| `MOTOR_ORIG` | numeric | optional | `1070` | Observed motorized count candidate |
| `DTVW` | numeric | optional | `1300` | Daily/period count candidate |
| `CARS_LEFT`, `CARS_RIGHT`, `CARS_STRAIGHT` | numeric | optional | `10` | Deferred turn-count candidates |
| `CARS_PROJ`, `HVG_PROJ`, `MOTOR_PROJ` | numeric | optional | `1000` | Deferred projected count values |
| `geometry` | point | optional | `POINT (...)` | Diagnostic location |

## Directional Semantics

`TSYSSET`, `TYPENO`, `LC`, `LENGTH`, `V0PRT`, `CAPPRT`, and `T0PRT` describe the AB direction from
`FROMNODENO` to `TONODENO`. Their `R_*` counterparts describe the BA direction and SHALL NOT be copied from the AB
fields when missing. Empty `TSYSSET` or `R_TSYSSET` means that direction is unavailable for private-traffic import.

`TYPENO` and `LC` are mapping candidates. The default mapping uses `LC` first, then `TYPENO`, and preserves the original
values in diagnostics. `R_TYPENO` and `R_LC` are recognized as reverse-direction metadata; v1 does not split one VISUM
link into two AequilibraE links solely because the reverse class differs.

## Deferred Layers

Known public-transport and context layers such as `stop`, `stoppoint`, `stop_point`, `lineroute`, `line_route`,
`ptline`, and OD matrix files are recognized and reported as deferred. They are not imported into private-traffic
network tables in v1.

## Checkpoint A, VISUM Semantics

Checkpoint A is approved for implementation against the compact fixture contract above. The main implementation risk is
that real VISUM exports may include agency-specific field aliases or units; those should be handled through explicit
mapping overrides rather than runtime inference.
