'''
Just changing eval_dir and coloropt will change the target txt & color option
'''

import pandas as pd
import plotly.express as px


def saveplot(df, coloropt, path, axis, showflag=False):
    # , selectdopt='posscreen_negexpnoexp'):
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    ax, ay = axis[0], axis[1]
    paramopt_lis = df[coloropt].values.tolist()
    paramopt_lis = list(set(paramopt_lis))
    revflag = False if coloropt == 'Data Update Type' else True
    paramopt_lis = sorted(paramopt_lis, reverse=revflag)

    fig = px.scatter(
        df, x=ax, y=ay, color=coloropt, color_discrete_sequence=colors, category_orders={coloropt: paramopt_lis}, hover_data=["Option"])

    for i, popt in enumerate(paramopt_lis):
        df_new = df[df[coloropt] == popt]
        x0, x1, y0, y1 = calc_coord(df_new, ax, ay)

        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=x0, y0=y0,
                      x1=x1, y1=y1,
                      opacity=0.4,
                      fillcolor=colors[i],
                      line_color=colors[i],)
    fig.update_layout(
        font=dict(
            size=15,
        )
    )

    fig.write_image(path)

    if showflag:
        fig.show()


def calc_coord(df_new, ax, ay):
    xvar = df_new[ax].std()
    xave = df_new[ax].mean()
    yvar = df_new[ay].std()
    yave = df_new[ay].mean()
    x0 = xave-xvar
    x1 = xave+xvar
    y0 = yave-yvar
    y1 = yave+yvar
    return x0, x1, y0, y1


def make_2dscatterplot(in_path, coloropt, out_path, axis):
    '''
    main method for making three 2d scatter plot
    '''
    df_tmp = pd.read_csv(in_path, sep='\t')
    df = df_tmp.replace('-', 0.0)
    df[coloropt] = df[coloropt].astype(str)
    df = df.iloc[:, :-1]
    saveplot(df, coloropt, out_path, axis)
    return df


def allopt_plot(df_dic, pngfile_list, axislis):
    '''
    DEFAULT: This will not run
    main method to make plot for all options
    1.concatenate dataframe of all option
    2.make plot for all options
    '''
    df_concat = pd.DataFrame()

    for k, df in df_dic.items():
        df['DataOption'] = k
        if df_concat.empty:
            df_concat = df
        else:
            df_concat = pd.concat([df_concat, df])

    for file, axis in zip(pngfile_list, axislis):
        saveplot(df_concat, 'DataOption', './'+file, axis, 'ALL')


def analyse_main(coloropt, summary_file, eval_dir, allplot=False):
    pngfile_list = ["prediction-diversity_"+coloropt+".png",
                    "prediction-property_"+coloropt+".png", "property-diversity_"+coloropt+".png"]
    axislis = [['Prediction', 'Diversity'], [
        'Prediction', 'Property'], ['Property', 'Diversity']]
    # whichopt: ["ut", "fe", "fp", "mp", "rev"]

    df_dic = {}

    for file, axis in zip(pngfile_list, axislis):
        df = make_2dscatterplot(
            eval_dir+summary_file, coloropt, eval_dir+file, axis)
    # df_dic[dopt] = df

    if allplot == True:  # Default:False
        allopt_plot(df_dic, pngfile_list, axislis)
