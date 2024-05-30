see_ai_labels = [
    (0,"angiodysplasia"),
    (1,"erosion"),
    (2,"stenosis"),
    (3,"lymphangiectasia"),
    (4,"lymph follicle"),
    (5,"SMT"),
    (6,"polyp-like"),
    (7,"bleeding"),
    (8,"diverticulum"),
    (9,"erythema"),
    (10,"foreign body"),
    (11,"vein"),
    ]

path_annotations = "../data/see_ai/SEE_AI_project_all_txt/SEE_AI_project_all_txt/"
path_images = "../data/see_ai/SEE_AI_project_all_images/SEE_AI_project_all_images/"

"""

This is how an example .txt looks:

image00012.txt:

1 0.497552 0.364138 0.230455 0.214136
1 0.693717 0.293715 0.236420 0.224914
1 0.527213 0.850271 0.630343 0.170283
9 0.531747 0.498954 0.876187 0.889307

This meas that image 12 should be copied into the folder corresponding to 1 -> erosion and 9 -> erythma.
"""