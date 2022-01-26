import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--last_ep", type=int, default=10,
                    help="")
# parser.add_argument("--inputopt",  action='store_true',
#                     help="choose whether or not you want to input the name of the option. Default:False, place --inputopt if you want it to be True.")
parser.add_argument("--allepoch",  action='store_true',
                    help="choose whether or not you want to evaluate all epoch's data. Default:False, place --allepoch if you want it to be True.")
parser.add_argument("--norealdata",  action='store_false',
                    help="choose whether or not you want to. Default:True, place --norealdata if you want it to be False.")
opt = parser.parse_args()

summary_file = "all.txt"
rank_file = "option_ranking.txt"
options = ["Epoch", "Data Update Type", "Feedback Epoch"]


def ranking(opt_list, rank_dic, rank_output_path):
    for i, option in enumerate(opt_list):
        if option in rank_dic.keys():
            rank_dic[option] += len(opt_list)-i
        else:
            rank_dic[option] = len(opt_list)-i
    sorted_rank_list = sorted(
        rank_dic.items(), key=lambda x: x[1], reverse=True)

    with open(rank_output_path, 'w') as f:
        header = options + ['Score'] + ['\n']
        f.write('\t'.join(header))
        for option, score in sorted_rank_list:
            f.write('\t'.join(option.split('_') + [str(score), '\n']))

    return rank_dic


def make_plot_dic(opt_list, num_list, plot_dic):
    for i, option in enumerate(opt_list):
        option = option.replace('_None', '')  # kari: deleting revNone
        if option in plot_dic.keys():
            plot_dic[option] += [num_list[i]]
        else:
            plot_dic[option] = [num_list[i]]

    return plot_dic


def make_summary(plot_dic, task_list, eval_dir):
    with open(eval_dir+summary_file, 'w') as f:
        # new_options = options[:-1]  # kari: deleting revNone
        header = ['Option'] + options + task_list + ['\n']
        f.write('\t'.join(header))
        for k, v in plot_dic.items():
            row = [k] + k.split('_') + [str(val) for val in v]
            f.write('\t'.join(row))
            f.write('\n')


def make_ranking_summary_plot(eval_dir, pred_file, prop_file, stdev_file):
    option_num = len(options)
    rank_dic = {}
    plot_dic = {}
    task_list = []

    path_dic = {'Prediction': eval_dir + pred_file,
                'Diversity': eval_dir+stdev_file, 'Property': eval_dir+prop_file}
    rank_output_path = eval_dir + rank_file

    for task, path in path_dic.items():
        task_list.append(task)
        with open(path) as f:
            header = f.readline()
            opt_list = []
            num_list = []

            for line in f:
                if not (('positive' in line) or ('negative' in line)):
                    row_list = line.split('\t')
                    opt_list.append("_".join(row_list[:option_num]))
                    num_list.append(float(row_list[len(row_list)-1][:-1]))

            rank_dic = ranking(opt_list, rank_dic, rank_output_path)
            plot_dic = make_plot_dic(opt_list, num_list, plot_dic)

            if len(task_list) == 3:
                make_summary(plot_dic, task_list, eval_dir)


def summary_main(eval_dir, pred_file, prop_file, stdev_file):
    make_ranking_summary_plot(eval_dir, pred_file, prop_file, stdev_file)
