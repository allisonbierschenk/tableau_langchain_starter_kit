# Tableau Dashboard Extension

When you use this app as a **dashboard extension** (embed in a Tableau dashboard via the `.trex` manifest), the Extension API is used to detect data sources from the dashboard.

## Required: Tableau Extensions API script

The HTML loads `static/tableau.extensions.1.latest.min.js`. If that file is missing, the app runs in **standalone mode** (no extension UI, no datasource detection).

To enable extension mode:

1. Download the [Tableau Extensions API SDK](https://github.com/tableau/extensions-api).
2. Copy `tableau.extensions.1.latest.min.js` (from the SDK’s `lib` or similar folder) into this project’s `static/` directory.

After that, when the app is loaded inside a Tableau dashboard, it will:

- Initialize the Extensions API and scan the dashboard’s worksheets for data sources
- POST the datasource map to `/datasources` so the backend can use it
- Show the floating chat icon and extension UI (resize, close)

When the app is opened **outside** Tableau (e.g. direct URL), it runs in standalone mode and does not require the Extension API.
