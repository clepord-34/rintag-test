import rintag

rintag.enable_passive_feedback(threshold=0.75, log_oov=True)

result = rintag.tag("Mayna, usaya na ding lamisa ta magkaun na kita")
print(result)

feedback = rintag.get_passive_feedback()
print(feedback)

rintag.export_passive_feedback("hitl_showcase/output/rintag_feedback_t.csv")