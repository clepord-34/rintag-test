import rintag

rintag.enable_passive_feedback(threshold=0.75, log_oov=True)

result = rintag.tag("pero mauungot si 'uya kin minsan ta diri ko isusulit a mga paborito niyang ngayamnan sadto kawon sabi niya")
print(result)

feedback = rintag.get_passive_feedback()
print(feedback)

rintag.export_passive_feedback("rintag_feedback_t.csv")