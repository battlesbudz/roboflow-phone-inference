# Product Receiving — Scope

Scope doc for the Receiving Mode feature in the phone app. Scope only — no
app code changes in this commit.

## Problem

When a shipment arrives at a dispensary, staff have to verify every item
against the shipping manifest, check labels and packaging, count units, catch
what is missing, wrong, or damaged, and log the accepted product into the
state seed-to-sale tracking system. Today that is manual, slow, and
error-prone. A missed mismatch means inventory that does not match the books,
and in cannabis that is a compliance problem, not just a bookkeeping one.

## Receiving workflow as it exists today (NY dispensary)

Verified from NY OCM / Metrc guidance (see Sources):

1. Shipment arrives with an invoice and a shipping manifest. The manifest is
   transmitted to OCM and the recipient before transport. It lists product
   name, item category, weight or count per package tag, unique package IDs,
   origin and destination license numbers, driver, and vehicle.
2. Staff verify the manifest against the physical shipment: product name,
   item category, weight/count per package tag, package IDs.
3. Visual inspection: damage, tampering, seals, labels, warning statements,
   cannabinoid profile, ingredients, product name, facility name and license.
4. Decision per shipment: accept, fully reject, or partially reject.
   Rejections go back on a new manifest to the sender.
5. Accepted product: create a new product entry in the inventory tracking
   system, new UID, supplier license noted, lab test results entered.
6. Damaged or defective units: quarantine, photo evidence, discrepancy
   record. Good and defective units are processed separately so defective
   units never land in available stock.

NY-specific rules that shape this feature:

- NY uses Metrc as the state seed-to-sale system. Metrc acquired
  BioTrack, so older BioTrack references in the industry now point at
  Metrc. Retail dispensary integration deadline was October 1, 2025.
- Tracking must be real-time and electronic. Paper records or spreadsheets
  are not compliant. Entering inventory movement after the fact is not
  compliant.
- Transfer flow is Invoice -> Manifest -> Transfer.

## Where vision fits

Vision can help with:

- Reading package labels, barcodes, QR codes: product name, batch, package
  UID. (OCR + barcode scanning.)
- Counting units in a case or tote and comparing the count to the manifest.
- Flagging mismatches: label vs manifest, count vs manifest, missing item.
- Capturing photo evidence of damage for the discrepancy record.
- Producing a per-item receiving checklist: match / mismatch / damaged,
  plus a session summary report.

Vision cannot / will not do:

- Accept or reject a shipment on its own. A human confirms every
  accept/reject decision.
- Write to Metrc or any seed-to-sale system. In v1 the human enters the
  decision in their own system. Any future direct integration would need
  authorized integrator status — an open question, not a v1 feature.
- Judge product quality, potency, or lab results. It reads what the label
  says. It does not test the product.
- Identify people. No demographics, no social-class inference, no
  race/gender/age anywhere in this feature.

## Feature: Receiving Mode (v1)

A new mode in the existing phone app, alongside Live Tracking. Same app,
same camera pipeline, not a new app.

- Point the phone camera (or watch the store RTSP camera) at the incoming
  shipment.
- Scan each package: read the label + package UID, count units.
- Compare against the manifest (staff-provided — format is an open question).
- Show per item: match / mismatch / damaged (staff taps damaged + photo).
- End of session: summary report (accepted list, flagged list, photos,
  timestamps) saved on the device. CSV export for Sheets / Excel.
- The human confirms accept/reject per item and enters it in their own
  tracking system.

## Data sources

- Phone camera: build, test, and demo.
- RTSP store cameras: production, same architecture as the visitor-counting
  mode (phone watches the store's existing camera streams over WiFi).
- Manifest: staff-provided. Photo of paper manifest, CSV export, or manual
  entry — open question.

## Success criteria

- A receiving session on a real shipment completes end to end with a human
  confirming each item.
- Mismatch detection catches a planted wrong-count / wrong-label case in
  testing.
- The session report + CSV export are usable by the person doing intake.
- Zero writes to any state tracking system in v1.

## Open questions (need Battles's input)

1. Do we have a real store / receiving area to test in, and who does intake
   there?
2. Manifest format: photo of the paper manifest, CSV export from their
   system, or manual entry?
3. Metrc: do we ever aim for authorized integrator status, or stay
   human-in-the-loop permanently?
4. Does receiving happen at a table (phone on a stand) or handheld?
5. What barcode types are on NY cannabis packaging (1D, QR, DataMatrix)?
6. Does the store's receiving area have a camera with a usable view of the
   intake table, or is the phone camera the only view?

## Verified vs assumed

Verified from public sources: NY uses Metrc; manifest contents and
pre-transport transmission; accept / fully reject / partially reject flow;
real-time electronic tracking requirement; Invoice -> Manifest -> Transfer.

Assumed, to confirm with a real operator: exact per-store SOP details
(barcode scan per unit vs count cases), barcode types on NY packaging,
whether a phone at the receiving table is acceptable workflow, and whether
OCR is reliable on real cannabis package labels. None of these are
presented as facts above.

## Sources

- NY OCM seed-to-sale bulletins: cannabis.ny.gov (STS bulletins, Metrc
  integration timeline and requirements)
- NY OCM Metrc inventory tracking FAQ: cannabis.ny.gov/inventory-tracking-FAQ
- NY cannabis guidance breakdown (manifest requirements): Cannabis Law Report
- General receiving / quarantine SOP patterns: pharmacy and warehouse
  receiving checklists (used as analogy only — not cannabis-specific)
