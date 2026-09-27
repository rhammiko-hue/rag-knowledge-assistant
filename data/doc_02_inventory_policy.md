# Inventory Management Policy

## Safety Stock
Safety stock is calculated as average daily demand multiplied by the desired coverage in days, plus a variability buffer based on demand standard deviation. Standard coverage targets are 7 days for A-class SKUs, 14 days for B-class, and 21 days for C-class items.

## Reorder Points
The reorder point equals average daily demand times supplier lead time in days, plus safety stock. Purchase orders are triggered automatically when on-hand inventory falls to the reorder point.

## ABC Classification
SKUs are classified annually by revenue contribution: A items represent roughly 80 percent of revenue, B items 15 percent, and C items 5 percent. Classification drives counting frequency, safety stock targets, and review cadence.

## Obsolete Inventory
Inventory with no movement for 180 days is flagged for review. The planning team decides whether to discount, return to vendor, or write off the stock each quarter.
