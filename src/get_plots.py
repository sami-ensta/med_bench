import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sns.set_style('whitegrid')
sns.set_context('talk', font_scale=2.35)





output_path = 'results/simulations/2023figures'





res_df = pd.read_csv('results/20230725_big_table_basic_simu.csv', sep='\t')




res_df.head() 







res_df = res_df.assign(med_setting='prop_low_no_overlap_violation')
res_df.loc[(res_df.wt_list<=2)&(res_df.wm_list>=5), 'med_setting'] = 'prop_high_no_overlap_violation'
res_df.loc[(res_df.wt_list>=1)&(res_df.wm_list<=2), 'med_setting'] = 'prop_high_strong_overlap_violation'
all_configs = ['prop_low_no_overlap_violation',
               'prop_high_no_overlap_violation',
               'prop_high_strong_overlap_violation']









nice_names_dict = \
{"coefficient_product"                  : "Coefficient product",
"DML_huber"                             : "medDML",
"med_dml_reg_not_normalized"    : "med dml (not normalized)",
"med_dml_reg_forest"                : "med dml (forest)",
"med_dml_reg_forest_calibration"    : "med dml (forest & calibration)",
"med_dml_reg_forest_calibration_cf" : "med dml (forest & calibration & cross-fitting)",
"med_dml_reg_forest_cf"             : "med dml (forest & cross-fitting)",
"med_dml_noreg"                 : "med dml (no regularization)",
"med_dml_noreg_cf"              : "med dml (no regularization & cross-fitting)",
"med_dml_reg"                   : "med dml (regularization)",
"med_dml_reg_calibration"       : "med dml (regularization & calibration)",
"med_dml_reg_calibration_cf"    : "med dml (regularization & calibration & cross-fitting)",
"med_dml_reg_cf"                : "med dml (regularization & cross-fitting)",
"multiply_robust_forest"                : "multiply robust (forest)",
"multiply_robust_forest_calibration"    : "multiply robust (forest & calibration)",
"multiply_robust_forest_calibration_cf" : "multiply robust (forest & calibration & cross-fitting)",
"multiply_robust_forest_cf"             : "multiply robust (forest & cross-fitting)",
"multiply_robust_noreg"                 : "multiply robust (no regularization)",
"multiply_robust_noreg_cf"              : "multiply robust (no regularization & cross-fitting)",
"multiply_robust_reg"                   : "multiply robust (regularization)",
"multiply_robust_reg_calibration"       : "multiply robust (regularization & calibration)",
"multiply_robust_reg_calibration_cf"    : "multiply robust (regularization & calibration & cross-fitting)",
"multiply_robust_reg_cf"                : "multiply robust (regularization & cross-fitting)"}
res_df = res_df.assign(nice_estimator=res_df.estimator.map(nice_names_dict))


res_df = res_df.assign(forest=res_df.nice_estimator.str.contains('forest'))
res_df = res_df.assign(cross_fitting=res_df.nice_estimator.str.contains('cross-fitting'))
res_df = res_df.assign(regularization=res_df.nice_estimator.str.contains('\(regularization'))
res_df = res_df.assign(base_estimator=res_df.nice_estimator.str.split(' \(').str.get(0))











res_df = res_df.assign(total_effect_error=(res_df.total_effect-res_df.total)/res_df.total,
                       direct_effect_error=(res_df.direct_control_effect-res_df.direct_0)/res_df.direct_0,
                       indirect_effect_error=(res_df.indirect_treated_effect-res_df.indirect_1)/res_df.indirect_1)







res_df[res_df.type_m=='continuous'].dim_m.value_counts()
res_df = res_df.assign(configuration = "")
res_df.loc[(res_df.type_m=='binary'), 'configuration'] = '1D binary'
res_df.loc[(res_df.type_m=='continuous')&(res_df.dim_m==1), 'configuration'] = '1D continuous'
res_df.loc[(res_df.type_m=='continuous')&(res_df.dim_m==5), 'configuration'] = '5D continuous'






res_df = res_df.assign(setting_name = "")
res_df.loc[(res_df.m_misspec==False)&(res_df.y_misspec==False), 'setting_name'] = "linear"
res_df.loc[(res_df.m_misspec==True)&(res_df.y_misspec==False), 'setting_name'] = "misspecification M"
res_df.loc[(res_df.m_misspec==False)&(res_df.y_misspec==True), 'setting_name'] = "misspecification Y"
res_df.loc[(res_df.m_misspec==True)&(res_df.y_misspec==True), 'setting_name'] = "misspecification M & Y"
setting_order = ["linear", "misspecification M", "misspecification Y", "misspecification M & Y"]








res_df.estimator.value_counts()





most_basic_estimators = ['coefficient_product','multiply_robust_noreg','DML_huber', 'med_dml_noreg']

robust_estimators = [
"multiply_robust_forest",
"multiply_robust_forest_calibration",
"multiply_robust_forest_calibration_cf",
"multiply_robust_forest_cf",
"multiply_robust_noreg",
"multiply_robust_noreg_cf",
"multiply_robust_reg",
"multiply_robust_reg_calibration",
"multiply_robust_reg_calibration_cf",
"multiply_robust_reg_cf",
]

dml_estimators = [
"DML_huber",
"med_dml_reg_not_normalized",
"med_dml_reg_forest",
"med_dml_reg_forest_calibration",
"med_dml_reg_forest_calibration_cf",
"med_dml_reg_forest_cf",
"med_dml_noreg",
"med_dml_noreg_cf",
"med_dml_reg",
"med_dml_reg_calibration",
"med_dml_reg_calibration_cf",
"med_dml_reg_cf",
]

all_estimators = [
"coefficient_product",
"DML_huber",
"med_dml_reg_not_normalized",
"med_dml_reg_forest",
"med_dml_reg_forest_calibration",
"med_dml_reg_forest_calibration_cf",
"med_dml_reg_forest_cf",
"med_dml_noreg",
"med_dml_noreg_cf",
"med_dml_reg",
"med_dml_reg_calibration",
"med_dml_reg_calibration_cf",
"med_dml_reg_cf",
"multiply_robust_forest",
"multiply_robust_forest_calibration",
"multiply_robust_forest_calibration_cf",
"multiply_robust_forest_cf",
"multiply_robust_noreg",
"multiply_robust_noreg_cf",
"multiply_robust_reg",
"multiply_robust_reg_calibration",
"multiply_robust_reg_calibration_cf",
"multiply_robust_reg_cf",
]





### NaN values : se corrige avec trim!=0
### Vient aussi des estimateurs non implémentés (multiply_robust multidim)
res_df.columns

is_continuous = np.any(res_df=="continuous",axis=1)
is_multiply = res_df['estimator'].str.contains("multiply", case=True, flags=0, na=None, regex=True)

nan_df = res_df[~ (is_continuous & is_multiply)]
nan_df = nan_df.drop('n_non_trimmed', axis=1)

nanobs = np.any(nan_df.isna(),axis=1)

estnan = nan_df.loc[nanobs]
estnan
estnan.estimator
len(estnan.estimator) # Number of Nan values
len(estnan.estimator)/len(res_df.estimator) # Proportion of NaN values

# Proportion des dataset NaN
len(estnan["mediated_prop"].unique())
len(estnan["mediated_prop"].unique())/5400

for _ in all_estimators: # Qui sont les NaN?
    print(_)
    print(_ in list(estnan.estimator))


res_df = res_df[res_df["duration"]<=10] # removing observations with aberant computation time




plotted_estimators = dml_estimators

for conf in all_configs:
    for effect in ['direct', 'indirect', 'total']:
        g = sns.catplot(x="n", y="{}_effect_error".format(effect), hue="nice_estimator",
                col="setting_name", row='configuration',
                data=res_df[(res_df.med_setting==conf)&(res_df.estimator.isin(plotted_estimators))],
                aspect=1.3, height=7,
                kind='point', sharey='row', margin_titles=True, dodge=0.6,
                capsize=0, join=False, col_order=setting_order, scale=2, errwidth=7, palette="cubehelix")
        for i in range(g.axes.shape[0]):
            for j in range(g.axes.shape[1]):
                true_val = 0
                g.axes[i, j].axhline(y=true_val, color="black", lw=4)
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        for ax in g.axes.ravel():
            ax.set_xlabel(ax.get_xlabel().replace('n', 'number of units'))
            ax.set_ylabel(ax.get_ylabel().replace("{}_effect_error".format(effect), 
                                                 '{} effect\nrelative error'.format(effect)))

        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        g = g.set_titles(row_template='{row_name}', col_template='{col_name}', weight='bold')
        hh = plt.setp(g._legend.get_texts(), fontsize=55)
        sns.move_legend(g, "lower left", bbox_to_anchor=(0.02, -0.1), ncol=3,
                        frameon=True, title="Estimator",
                        title_fontproperties={'weight':'bold'},
                        bbox_transform=g.fig.transFigure)

        g._legend.set_in_layout(in_layout=True)


        plt.savefig('{}/{}_{}_effect.pdf'.format(output_path, conf, effect),
                    bbox_extra_artists=(g._legend, g.fig), bbox_inches='tight')




for conf in all_configs:
    for effect in ['duration']:
        g = sns.catplot(x="n", y="{}".format(effect), hue="nice_estimator",
                col="setting_name", row='configuration',
                data=res_df[(res_df.med_setting==conf)&(res_df.estimator.isin(plotted_estimators))],
                aspect=1.3, height=7,
                kind='point', sharey='row', margin_titles=True, dodge=0.6,
                capsize=0, join=False, col_order=setting_order, scale=2, errwidth=7, palette="cubehelix")
        for i in range(g.axes.shape[0]):
            for j in range(g.axes.shape[1]):
                true_val = 0
                g.axes[i, j].axhline(y=true_val, color="black", lw=4)
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        for ax in g.axes.ravel():
            ax.set_xlabel(ax.get_xlabel().replace('n', 'number of units'))
            ax.set_ylabel(ax.get_ylabel().replace("{}".format(effect), 
                                                 'computation time (s)'.format(effect)))

        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        g = g.set_titles(row_template='{row_name}', col_template='{col_name}', weight='bold')
        hh = plt.setp(g._legend.get_texts(), fontsize=55)
        sns.move_legend(g, "lower left", bbox_to_anchor=(0.02, -0.1), ncol=3,
                        frameon=True, title="Estimator",
                        title_fontproperties={'weight':'bold'},
                        bbox_transform=g.fig.transFigure)

        g._legend.set_in_layout(in_layout=True)


        plt.savefig('{}/{}_{}.pdf'.format(output_path, conf, effect),
                    bbox_extra_artists=(g._legend, g.fig), bbox_inches='tight')
