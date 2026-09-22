from apex_decision import ApexClient

print(ApexClient().score("Rate the response", ["poor", "good", "excellent"], "The response was complete."))
