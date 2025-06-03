import numpy as np


def gather_iam_info(gtfile, root_path, valid_set):
    gt = []
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]
            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            #if level == 'word':
            line_name = pathlist[-2]
            del pathlist[-2]

            form_name = '-'.join(line_name.split('-')[:-1])

            #if (info[1] != 'ok') or (form_name not in valid_set):

            #if (level == 'word') and (info[1] != 'ok'):
            if info[1] != 'ok':
                continue

            if (form_name not in valid_set):
                #print(line_name)
                continue
            img_path = '/'.join(pathlist)
            transcr = ' '.join(info[8:])
            gt.append((img_path, transcr))
    return gt


if __name__ == "__main__":
    # check nb item per split
    root_path = "C:/Users/simcor/dev/data/IAM/origin/words/"
    gtfile = "C:/Users/simcor/dev/data/IAM/origin/ascii/words.txt"

    path_split_train = "C:/Users/simcor/dev/projects/HTR-best-practices/utils/aachen_iam_split/train.uttlist"
    valid_set = np.loadtxt(path_split_train, dtype=str)

    info_train = gather_iam_info(gtfile, root_path, valid_set)
    print("Train: " + str(len(info_train)))

    path_split_val = "C:/Users/simcor/dev/projects/HTR-best-practices/utils/aachen_iam_split/validation.uttlist"
    valid_set = np.loadtxt(path_split_val, dtype=str)

    info_val = gather_iam_info(gtfile, root_path, valid_set)
    print("Validation: " + str(len(info_val)))

    path_split_test = "C:/Users/simcor/dev/projects/HTR-best-practices/utils/aachen_iam_split/test.uttlist"
    valid_set = np.loadtxt(path_split_test, dtype=str)

    info_test = gather_iam_info(gtfile, root_path, valid_set)
    print("Test: " + str(len(info_test)))

