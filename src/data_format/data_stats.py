import glob
import math

from skimage import io


def print_stats_text_datasets_lines(dir_data):
    files = glob.glob(dir_data + '/**/*.txt', recursive=True)

    full_text = ""

    nb_total_letter = 0
    nb_total_words = 0

    min_len_txt = 999999
    max_len_txt = 0
    mean_len_txt = 0

    min_nb_words_txt = 999999
    max_nb_words_txt = 0
    mean_nb_words_txt = 0

    nb_items = 0

    dict_class_nb = {}
    dict_line_length = {}

    for one_file_label in files:
        label = ""
        with open(one_file_label, "r", encoding="utf-8") as file:
            label = file.readline()

        nb_total_letter += len(label)
        label_word = label.split(" ")

        nb_total_words += len(label_word)

        full_text += label

        for one_char in label:
            if one_char in dict_class_nb:
                dict_class_nb[one_char] += 1
            else:
                dict_class_nb[one_char] = 1

        len_txt = len(label)
        if len_txt < min_len_txt:
            min_len_txt = len_txt
        if len_txt > max_len_txt:
            max_len_txt = len_txt

        mean_len_txt += len_txt

        if len_txt in dict_line_length:
            dict_line_length[len_txt] += 1
        else:
            dict_line_length[len_txt] = 1

        nb_words = len(label_word)

        if nb_words < min_nb_words_txt:
            min_nb_words_txt = nb_words
        if nb_words > max_nb_words_txt:
            max_nb_words_txt = nb_words

        mean_nb_words_txt += nb_words

        nb_items += 1

    mean_len_txt /= nb_items
    mean_nb_words_txt /= nb_items

    print("nb_items :" + str(nb_items))
    print()
    print("min_len_txt :" + str(min_len_txt))
    print("max_len_txt :" + str(max_len_txt))
    print("mean_w :" + str(mean_len_txt))
    print()
    print("min_nb_words_txt :" + str(min_nb_words_txt))
    print("max_nb_words_txt :" + str(max_nb_words_txt))
    print("mean_nb_words_txt :" + str(mean_nb_words_txt))
    print()
    myKeys = list(dict_line_length.keys())
    myKeys.sort()
    sorted_dict = {i: dict_line_length[i] for i in myKeys}
    print("dict_line_length:")
    print(sorted_dict)

    print()
    myKeys = list(dict_class_nb.keys())
    myKeys.sort()
    sorted_dict = {i: dict_class_nb[i] for i in myKeys}
    print("dict_class_nb :")
    print(sorted_dict)
    print("nb_total_letter :" + str(nb_total_letter))
    print("nb_total_words :" + str(nb_total_words))
    print()


def print_stats_img_datasets(dir_data, extension="png"):
    if isinstance(extension, list):
        files = []

        for one_e in extension:
            one_ext_file = glob.glob(dir_data + '/**/*.' + one_e, recursive=True)
            files.extend(one_ext_file)
    else:
        files = glob.glob(dir_data + '/**/*.' + extension, recursive=True)

    min_w = 999999
    max_w = 0
    mean_w = 0

    dict_width = {}

    min_h = 999999
    max_h = 0
    mean_h = 0

    dict_height = {}

    nb_items = 0

    for one_file in files:
        nb_items += 1
        img = io.imread(one_file)

        if len(img.shape) == 2:
            h, w = img.shape
        elif len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            print("Format image not supported -> continue")

        if h < min_h:
            min_h = h
        if h > max_h:
            max_h = h

        if h in dict_height:
            dict_height[h] += 1
        else:
            dict_height[h] = 1

        mean_h += h

        if w < min_w:
            min_w = w
        if w > max_w:
            max_w = w

        if w in dict_width:
            dict_width[w] += 1
        else:
            dict_width[w] = 1

        mean_w += w

    mean_w /= nb_items
    mean_h /= nb_items

    myKeys = list(dict_width.keys())
    myKeys.sort()
    sorted_dict_w = {i: dict_width[i] for i in myKeys}

    print("dict_width :")
    print(sorted_dict_w)
    print("min_w :" + str(min_w))
    print("max_w :" + str(max_w))
    print("mean_w :" + str(mean_w))
    print()
    myKeys = list(dict_height.keys())
    myKeys.sort()
    sorted_dict_h = {i: dict_height[i] for i in myKeys}

    print("dict_height :")
    print(sorted_dict_h)
    print("min_h :" + str(min_h))
    print("max_h :" + str(max_h))
    print("mean_h :" + str(mean_h))


if __name__ == "__main__":
    path_db = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1/train/images/"
    
    print_stats_img_datasets(path_db)

