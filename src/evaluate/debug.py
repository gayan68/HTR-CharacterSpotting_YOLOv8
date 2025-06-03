
for iou_v in [x / 100.0 for x in range(35, 65, 5)]:  # range(0.35, 0.65, 0.05):
    for conf_v in [x / 100.0 for x in range(20, 40, 5)]:  # range(0.2, 0.4, 0.05):
        print("iou: " + str(iou_v))
        print("conf_v: " + str(conf_v))
