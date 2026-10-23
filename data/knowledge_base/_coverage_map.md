# HelpDeskAI v1 Coverage Map

All content below describes synthetic Meridian Works material created for a portfolio project. It is not an official policy.

## Document inventory

- **File:** `wifi-connection-troubleshooting-guide.txt`
  **Document ID:** `wifi-connection-troubleshooting-guide`; **Format:** TXT; **Category:** `network_wifi` (Network and Wi-Fi); **Word count:** 591.
  **Summary:** Diagnoses employee wireless connection failures on Meridian Works' fictional office network.
  **Key facts:** Network: Meridian-Staff; guest network: Meridian-Visitor; DHCP renewal wait: 90 seconds; support window: 15 minutes; code WIFI-214.
  **Error code defined:** `WIFI-214` — definition is in this document only.
  **Escalation rule:** Raise a medium-priority ticket if one device remains disconnected after the steps above or the issue lasts more than 15 minutes. Raise high priority when at least five users in one work area lose staff Wi-Fi at the same time. Report suspected electrical damage, heat, or a burning smell through the safety route immediately rather than continuing network tests.

- **File:** `remote-access-vpn-guide.txt`
  **Document ID:** `remote-access-vpn-guide`; **Format:** TXT; **Category:** `network_wifi` (Network and Wi-Fi); **Word count:** 566.
  **Summary:** Separates remote tunnel failures from local Wi-Fi problems and explains escalation.
  **Key facts:** Client: Meridian Tunnel; retry wait: 60 seconds; session idle limit: 30 minutes; code VPN-612.
  **Error code defined:** `VPN-612` — definition is in this document only.
  **Escalation rule:** Use medium priority when one user cannot establish a tunnel after two attempts. Use high priority if a team-wide remote access outage blocks scheduled work. Use critical only when a confirmed remote-access failure is part of a broader incident with immediate business-safety impact, and follow the incident coordinator's instructions.

- **File:** `laptop-docking-and-display-guide.pdf`
  **Document ID:** `laptop-docking-and-display-guide`; **Format:** PDF; **Category:** `hardware_peripherals` (Hardware and peripherals); **Word count:** 556.
  **Summary:** Troubleshoots docks, monitors, cables, and external display detection.
  **Key facts:** Approved dock family: Meridian Dock D4; power reset: 20 seconds; monitor detection wait: 45 seconds; code DISP-207.
  **Error code defined:** `DISP-207` — definition is in this document only.
  **Escalation rule:** Raise low priority for a single display issue with a usable laptop screen. Raise medium priority if the dock prevents normal work and the checks fail. Raise high priority if a shared workstation used for a time-critical operational function is unavailable. Stop using the equipment and report immediately if a connector is hot, sparking, or smells burnt.

- **File:** `swollen-battery-device-safety.docx`
  **Document ID:** `swollen-battery-device-safety`; **Format:** DOCX; **Category:** `hardware_peripherals` (Hardware and peripherals); **Word count:** 580.
  **Summary:** Provides immediate stop-use instructions for a swollen, hot, or damaged portable-device battery.
  **Key facts:** Safe response boundary: 2 metres if practical; do not recharge; safety desk response target: 5 minutes; code BAT-901.
  **Error code defined:** `BAT-901` — definition is in this document only.
  **Escalation rule:** This is an immediate safety escalation, not a routine repair. Active smoke, hissing, fire, or rapidly increasing heat requires the site's emergency response route now. A visibly swollen battery without active heat still requires a high-priority safety ticket and the device must remain out of service.

- **File:** `software-installation-and-update-guide.pdf`
  **Document ID:** `software-installation-and-update-guide`; **Format:** PDF; **Category:** `software_applications` (Software and applications); **Word count:** 546.
  **Summary:** Explains the fictional software catalogue, installation checks, and update troubleshooting.
  **Key facts:** Portal: Meridian App Shelf; standard catalogue sync: 10 minutes; approval target: 2 business days; code APP-318.
  **Error code defined:** `APP-318` — definition is in this document only.
  **Escalation rule:** Raise low priority for a non-urgent catalogue question. Raise medium priority if an approved application is unavailable to one user after the synchronization check. Raise high priority when a required application fails for a group with an active operational deadline. Do not elevate a request merely because approval is pending within its stated target.

- **File:** `application-crash-and-cache-guide.docx`
  **Document ID:** `application-crash-and-cache-guide`; **Format:** DOCX; **Category:** `software_applications` (Software and applications); **Word count:** 537.
  **Summary:** Covers application launch failures, safe cache cleanup, and application-owner escalation.
  **Key facts:** App: Meridian DeskView; cache retention: 7 days; retry interval: 30 seconds; code APP-381.
  **Error code defined:** `APP-381` — definition is in this document only.
  **Escalation rule:** Raise low priority for an intermittent cosmetic issue with a workaround. Raise medium priority if the application crashes repeatedly after one restart and a cache refresh. Raise high priority if multiple users cannot access a work-critical function. Include the exact time and reproducible steps; do not attach customer or confidential records unless an approved secure channel is specified.

- **File:** `email-delivery-and-sync-guide.pdf`
  **Document ID:** `email-delivery-and-sync-guide`; **Format:** PDF; **Category:** `email_collaboration` (Email and collaboration tools); **Word count:** 580.
  **Summary:** Diagnoses delayed messages, mail client sync, and delivery failures.
  **Key facts:** Webmail: Meridian Mailroom; sync refresh: 5 minutes; recoverable deleted items: 14 days; code MAIL-422.
  **Error code defined:** `MAIL-422` — definition is in this document only.
  **Escalation rule:** Use low priority for one delayed message with a working webmail workaround. Use medium priority if synchronization remains broken after the 5-minute refresh or outbound messages are blocked. Use high priority for a widespread delivery failure affecting a team. A suspected account compromise must be reported through the account-security route immediately.

- **File:** `shared-mailbox-and-distribution-list-guide.docx`
  **Document ID:** `shared-mailbox-and-distribution-list-guide`; **Format:** DOCX; **Category:** `email_collaboration` (Email and collaboration tools); **Word count:** 560.
  **Summary:** Distinguishes shared mailbox access from distribution-list membership and approval.
  **Key facts:** Portal: Meridian Access Desk; owner approval target: 1 business day; membership sync: 20 minutes; code COL-205.
  **Error code defined:** `COL-205` — definition is in this document only.
  **Escalation rule:** Raise low priority for a question about the resource type. Raise medium priority when an approved change has not appeared after 20 minutes. Raise high priority only when the access failure blocks a time-critical team function and the owner has confirmed the request. Unauthorized access concerns go to the security route immediately.

- **File:** `account-lockout-and-signin-guide.pdf`
  **Document ID:** `account-lockout-and-signin-guide`; **Format:** PDF; **Category:** `account_access` (Account access and permissions); **Word count:** 525.
  **Summary:** Explains failed sign-in checks, temporary lockouts, and safe account recovery.
  **Key facts:** Identity portal: Meridian Identity Gate; lockout duration: 15 minutes; recovery link expiry: 10 minutes; code AUTH-403.
  **Error code defined:** `AUTH-403` — definition is in this document only.
  **Escalation rule:** Use medium priority when a single user remains locked out after the 15-minute wait and supported recovery. Use high priority if multiple accounts are locked unexpectedly or a suspected credential attack is underway. Report any unexpected verification prompt or suspected compromise immediately through the security route.

- **File:** `application-permission-request-guide.docx`
  **Document ID:** `application-permission-request-guide`; **Format:** DOCX; **Category:** `account_access` (Account access and permissions); **Word count:** 534.
  **Summary:** Separates successful sign-in from authorization failures and explains approval workflow.
  **Key facts:** Portal: Meridian Access Desk; manager approval target: 2 business days; role sync: 30 minutes; code ACC-403.
  **Error code defined:** `ACC-403` — definition is in this document only.
  **Escalation rule:** Raise low priority for a general role question. Raise medium priority when an approved role is still missing after 30 minutes. Raise high priority when an access denial blocks a confirmed time-critical task and the approval is complete. A denial by itself is not evidence of an account compromise.

- **File:** `printer-queue-and-paper-jam-guide.txt`
  **Document ID:** `printer-queue-and-paper-jam-guide`; **Format:** TXT; **Category:** `printing_office_equipment` (Printing and office equipment); **Word count:** 568.
  **Summary:** Covers office printer queues, safe jam clearing, and print-job escalation.
  **Key facts:** Print service: Meridian Print Lane; queue refresh: 2 minutes; job retention: 8 hours; code PRN-117.
  **Error code defined:** `PRN-117` — definition is in this document only.
  **Escalation rule:** Use low priority for one delayed print job with another printer available. Use medium priority if a shared printer remains unusable after the safe checks. Use high priority if a designated operational print station is unavailable for a time-critical process. Stop and escalate immediately if there is smoke, burning odor, sparking, or unusual heat.

- **File:** `scanner-and-copier-guide.pdf`
  **Document ID:** `scanner-and-copier-guide`; **Format:** PDF; **Category:** `printing_office_equipment` (Printing and office equipment); **Word count:** 559.
  **Summary:** Troubleshoots scan destinations, copier feeding, and document-output issues.
  **Key facts:** Scan destination: Meridian Drop Folder; file-size guidance: 20 MB; destination sync wait: 3 minutes; code SCN-228.
  **Error code defined:** `SCN-228` — definition is in this document only.
  **Escalation rule:** Use low priority for a one-off scan that can be completed on another approved device. Use medium priority when a shared scanner cannot deliver files after the destination refresh. Use high priority when a required time-sensitive workflow is blocked. Stop immediately if the copier smells burnt, sparks, or shows liquid damage.

- **File:** `water-leak-and-electrical-safety-guide.docx`
  **Document ID:** `water-leak-and-electrical-safety-guide`; **Format:** DOCX; **Category:** `facilities_maintenance` (Facilities and maintenance); **Word count:** 586.
  **Summary:** Sets immediate safety actions for leaks near outlets, panels, or powered equipment.
  **Key facts:** Keep-clear distance: 3 metres if practical; site safety acknowledgement target: 5 minutes; code SAFE-704.
  **Error code defined:** `SAFE-704` — definition is in this document only.
  **Escalation rule:** This is an immediate safety escalation. Water near electrical equipment, an electrical panel, or a powered fixture must be reported to Site Safety immediately; use the emergency route for sparking, smoke, or immediate danger. Do not downgrade it to a routine plumbing ticket until the authorized responder confirms the electrical risk is controlled.

- **File:** `burning-smell-and-power-fault-guide.txt`
  **Document ID:** `burning-smell-and-power-fault-guide`; **Format:** TXT; **Category:** `facilities_maintenance` (Facilities and maintenance); **Word count:** 558.
  **Summary:** Provides immediate response guidance for burning odors, sparks, hot outlets, and power faults.
  **Key facts:** Keep-clear distance: 2 metres if practical; safety acknowledgement target: 5 minutes; code ELEC-808.
  **Error code defined:** `ELEC-808` — definition is in this document only.
  **Escalation rule:** Immediate safety escalation is required for burning odor, sparks, crackling, smoke, or an unusually hot outlet. Use the emergency route for active smoke, flame, or immediate danger. Do not create a routine maintenance ticket as the only response and do not re-energize the equipment yourself.

- **File:** `general-faq-and-ticket-guidelines.txt`
  **Document ID:** `general-faq-and-ticket-guidelines`; **Format:** TXT; **Category:** `general` (General); **Word count:** 635.
  **Summary:** Explains how to submit useful tickets, choose priority, and track ownership without replacing specialist procedures.
  **Key facts:** Portal: Meridian Help Point; acknowledgement targets: low 1 business day, medium 4 business hours, high 30 minutes, critical 10 minutes; code TKT-101.
  **Error code defined:** `TKT-101` — definition is in this document only.
  **Escalation rule:** Meridian Help Point acknowledgement targets are low 1 business day, medium 4 business hours, high 30 minutes, and critical 10 minutes. These are acknowledgement targets, not resolution commitments. Any immediate physical hazard or suspected account compromise must be reported through its designated urgent route without waiting for portal acknowledgement.

## Confusable document pairs
- `wifi-connection-troubleshooting-guide.txt` and `remote-access-vpn-guide.txt`: local staff Wi-Fi versus remote tunnel connectivity are different layers.
- `laptop-docking-and-display-guide.pdf` and `swollen-battery-device-safety.docx`: routine display fault versus portable-device battery safety hazard.
- `swollen-battery-device-safety.docx` and `burning-smell-and-power-fault-guide.txt`: portable-device battery damage versus building electrical overheating/arcing.
- `software-installation-and-update-guide.pdf` and `application-crash-and-cache-guide.docx`: installation/catalogue rollback versus runtime launch/cache failure.
- `email-delivery-and-sync-guide.pdf` and `shared-mailbox-and-distribution-list-guide.docx`: message synchronization/delivery versus resource type and delegated access.
- `account-lockout-and-signin-guide.pdf` and `application-permission-request-guide.docx`: failed authentication/temporary lockout versus successful sign-in with missing authorization.
- `printer-queue-and-paper-jam-guide.txt` and `scanner-and-copier-guide.pdf`: print queue and paper output versus scan capture and network delivery.
- `water-leak-and-electrical-safety-guide.docx` and `burning-smell-and-power-fault-guide.txt`: water intrusion near electrical equipment versus electrical overheating or arcing.
- `general-faq-and-ticket-guidelines.txt` and every specialist guide: general ticket triage does not replace domain-specific troubleshooting or immediate hazard procedures.

## Intentional knowledge-base gaps
These topics are deliberately absent from all 15 documents and should remain unanswerable from this knowledge base:
- Payroll dates, salary slips, salary deductions, and compensation disputes.
- Leave balances, leave approval, annual leave, and sick leave.
- HR policies, employee relations, and disciplinary procedures.
- Recruitment status, interview feedback, and job applications.
- Employee benefits, insurance claims, and dependent coverage.
- Tax declarations, tax forms, and investment declarations.
- Expense reimbursement and business travel claims.
- Attendance corrections, biometric punch disputes, and shift swaps.
- Performance reviews, promotion criteria, and salary revision cycles.
- Employee personal data changes such as legal name or home address.
- Legal advice, contract interpretation, and employment-law disputes.
- Procurement budgets, vendor selection, and purchase-order approval thresholds.
- Physical security badge replacement, visitor pre-registration, and lost property.
- Cafeteria menus, food allergies, and meal subsidies.
- Shuttle routes, transport allowances, parking allocations, and commuting claims.
- Medical advice, diagnosis, medication, and occupational health assessments.
- Business strategy, confidential financial results, and customer account data.
- Passwords, credentials, recovery codes, private keys, or any secret value.

## Synthetic assumptions
- Meridian Works, all portals, networks, application names, team names, room labels, and diagnostic codes are fictional.
- Time targets are illustrative service targets for this dataset, not commitments by a real organization.
- The general document gives ticket acknowledgement targets; specialist documents give domain-specific escalation conditions.
- Each diagnostic code is defined in exactly one document. The code inventory here indexes that definition.