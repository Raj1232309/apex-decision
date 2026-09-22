from apex_decision import ApexClient

print(ApexClient().choice("Select priority", ["low", "medium", "high"], "The incident is urgent."))
