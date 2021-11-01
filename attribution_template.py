import pandas as pd
import numpy as np

pd.options.display.max_columns = 25


def find_last_touchpoint(t_row, touch_col_prepend_f, max_touches_f):
    for t in range(max_touches_f, 0, -1):
        t_val = t_row[touch_col_prepend_f + str(t)]
        try:
            if isinstance(t_val, str):
                return t_val
        except:
            None
    return None


def find_first_touchpoint(t_row, touch_col_prepend_f, max_touches_f):
    for t in range(1, max_touches_f + 1):
        t_val = t_row[touch_col_prepend_f + str(t)]
        try:
            if isinstance(t_val, str):
                return t_val
        except:
            None
    return None


def find_last_nondirect_touchpoint(t_row, touch_col_prepend_f, max_touches_f, direct_label_f):
    for t in range(max_touches_f, 0, -1):
        t_val = t_row[touch_col_prepend_f + str(t)]
        try:
            if isinstance(t_val, str) and t_val != direct_label_f:
                return t_val
        except:
            None
    return None


def assign_credit(t_row, cred_col_names_f, touch_col_names_f, cred_col_post_pend_f, model_type_f, first_weight_f=0.5, last_weight_f=0.5):
    # function assigns a credit to each relevant channel based on user specified model type, e.g. "last_touch_point", "first_touch_point", etc.
    t_dict = dict(zip(cred_col_names_f, [0]*len(cred_col_names_f)))

    if model_type_f == 'last_touch_point':
        # last
        t_dict.update({t_row['last_touch_point'] + cred_col_post_pend_f: 1})
        return t_dict
    elif model_type_f == 'first_touch_point':
        # first
        t_dict.update({t_row['first_touch_point'] + cred_col_post_pend_f: 1})
        return t_dict
    elif model_type_f == 'last_nondirect_touch_point':
        # last_non_direct
        try:
            t_dict.update({t_row['last_nondirect_touch_point'] + cred_col_post_pend_f: 1})
            return t_dict
        except TypeError:
            # case where there is no other channel
            t_dict.update({'direct' + cred_col_post_pend_f: 1})
            return t_dict
    elif (model_type_f == 'linear') or (model_type_f == 'position'):
        # linear and position based
        t_channels = [x for x in t_row[touch_col_names_f] if isinstance(x, str)]
        if model_type_f == 'linear':
            # linear weights
            t_weights = [1 / len(t_channels)] * len(t_channels)
        elif model_type_f == 'position':
            # position based weights (first and last specified, middle divided evenly)
            if len(t_channels) > 2:
                t_weights = [first_weight_f] + [(1 - (first_weight_f + last_weight_f)) / (len(t_channels) - 2)] * (len(t_channels) - 2) + [last_weight_f]
            elif len(t_channels)==1:
                t_weights = [1]
            else:
                t_weights = [first_weight_f] + [last_weight_f]

        t_weights = [x / sum(t_weights) for x in t_weights]     # ensure weights sum to 1
        for i in range(0, len(t_weights)):
            t_key = t_channels[i] + '_credit'
            t_value = t_dict[t_key] + t_weights[i]
            t_dict.update({t_key: t_value})
        return t_dict
    else:
        return t_dict


def get_attribution_by_channel(df_f, credit_col_postpend_f):
    allocated_conversions = df_f[cred_col_names].sum()
    n_allocated_conversions = df_f[cred_col_names].sum().sum()
    n_total_conversions = df_f.convert_TF.sum()
    if n_allocated_conversions != n_total_conversions:
        print('WARNING: allocation error. Sum of allocated conversions = %d. Sum of total conversions = %d' % (int(n_allocated_conversions), int(n_total_conversions)))

    channel_allocation_f = pd.Series(dict(zip([x.split(credit_col_postpend_f)[0] for x in allocated_conversions.keys()], list(allocated_conversions.array))))
    return channel_allocation_f


def calc_avg_CAC(channel_allocation_f, channel_spend_f):
    t_df = pd.DataFrame(channel_allocation_f)
    t_df.columns = ['channel_allocation']
    for t_ind, _ in t_df.iterrows():
        t_df.loc[t_ind, 'channel_spend'] = channel_spend_f[t_ind]

    t_df['CAC'] = t_df['channel_spend'] / t_df['channel_allocation']
    t_df['CAC'].replace(np.inf, 0, inplace=True)
    return t_df


def calc_marginal_CAC(n_conversions_low_tier, spend_low_tier, n_conversions_high_tier, spend_high_tier):
    ##### fill in this code to create the three variables in output dictionary
    marginal_spend = spend_high_tier - spend_low_tier
    marginal_conversions = n_conversions_high_tier - n_conversions_low_tier
    marginal_CAC = marginal_spend / marginal_conversions
    return {'marginal_conversions': marginal_conversions, 'marginal_spend': marginal_spend,
            'marginal_CAC': marginal_CAC}


# ----- Set parameters -----
touch_col_prepend = 'touch'
direct_label = 'direct'
first_weight = 0.4
last_weight = 0.4
cred_col_post_pend = '_credit'
select_model_types = ['last_touch_point', 'first_touch_point', 'last_nondirect_touch_point', 'linear', 'position']    # options are ['last_touch_point', 'first_touch_point', 'last_nondirect_touch_point', 'linear', 'position']
write_to_file = True

# ----- Import data -----
df = pd.read_pickle('attribution_allocation_student_data')
channel_spend = pd.read_pickle('channel_spend_student_data')

# ##### This data set is large. As you work through the code,
# you will find that executing some functions can take a long time (O(hours))
# I would recommend a technique for dealing with large data sets: randomly sample a small portion of it to work with.
# When you are satisfied that (1) the code executes end-to-end (2) the results look as you would expect (sanity check),
# then run on the full data set. Additionally, if you find that you need to run a time-consuming function repeatedly as
# an upstream input, consider saving down intermediate data structures so that you can load them into memory rather
# than having to process the data from the beginning. There is no code that *needs* to be added here,
# these are just best-practices suggestions.

# Only calculate the customers who converted with random sampling
# The first line could be changed into False to calculate the whole sample
if True:
    df = df[df['convert_TF'] == True]
    df = df.sample(frac=0.1, random_state=None, axis=0, replace=False)


# ----- Calculations -----
touch_col_names = [x for x in df.columns if x.find(touch_col_prepend) > -1]
max_touches = max([int(x.split(touch_col_prepend)[1]) for x in touch_col_names])

# total spending for all three tier experiments
channel_spend['total'] = dict()
for t_name, t in channel_spend.items():
    if t_name != 'total':
        for c in t.keys():
            try:
                channel_spend['total'][c] = channel_spend['total'][c] + t[c]
            except KeyError:
                channel_spend['total'].update({c: 0})

# ----- Format dataframe -----
# --- create credit columns
base_set = set()
[base_set.update(set(df[x].dropna().unique())) for x in touch_col_names]
cred_col_names = [x + '_credit' for x in base_set]
df = pd.concat([df, pd.DataFrame(data=0, columns=cred_col_names, index=df.index)], axis=1, ignore_index=False)

# --- identify key touch points
df['last_touch_point'] = df.apply(find_last_touchpoint, args=(touch_col_prepend, max_touches), axis=1)
df['first_touch_point'] = df.apply(find_first_touchpoint, args=(touch_col_prepend, max_touches), axis=1)
df['last_nondirect_touch_point'] = df.apply(find_last_nondirect_touchpoint, args=(touch_col_prepend, max_touches, direct_label,), axis=1)

# ----- RUN MODELS -----
CAC_dfs = dict()
for model_type in select_model_types:
    print('Processing model %s' % model_type)

    # ----- Run attribution model -----
    print('Running attribution model')
    df_convert = df.loc[df.convert_TF]  # only run calculation for conversion rows
    for t_ind, t_row in df_convert.iterrows():
        t_credit_dict = assign_credit(t_row, cred_col_names, touch_col_names, cred_col_post_pend, model_type, first_weight, last_weight)
        df.loc[t_ind, list(t_credit_dict.keys())] = list(t_credit_dict.values())  # add credit to original dataframe
    del df_convert, t_ind, t_row

    # ----- Calculate CAC -----
    print('Calculating average and marginal CAC')
    # --- Average CAC ---
    channel_allocation = get_attribution_by_channel(df, credit_col_postpend_f='_credit')
    df_CAC = calc_avg_CAC(channel_allocation_f=channel_allocation, channel_spend_f=channel_spend['total'])

    # --- Marginal CAC ---
    credit_cols = [x for x in df.columns if x.find('credit') > -1]
    df_CAC = pd.DataFrame(index=[x.split('_credit')[0] for x in credit_cols])
    base_col_names = ['marginal_conversions', 'marginal_spend', 'marginal_CAC']

    df_tier_sum = df[['tier']+credit_cols].groupby(['tier']).sum()
    df_tier_sum.columns = [x.split('_credit')[0] for x in df_tier_sum.columns]
    for t_tier in df_tier_sum.index:
        for t_channel in df_CAC.index:
            if t_tier > 1:
                n_conversions_low_tier = df_tier_sum.loc[t_tier - 1, t_channel]
                spend_low_tier = channel_spend['tier' + str(t_tier - 1)][t_channel]
                n_conversions_high_tier = df_tier_sum.loc[t_tier, t_channel]
                spend_high_tier = channel_spend['tier' + str(t_tier)][t_channel]
            else:
                n_conversions_low_tier = 0
                spend_low_tier = 0
                n_conversions_high_tier = df_tier_sum.loc[t_tier, t_channel]
                spend_high_tier = channel_spend['tier' + str(t_tier)][t_channel]

            t_df_CAC_colnames = [x + '_t' + str(t_tier) for x in base_col_names]
            print(n_conversions_low_tier)
            print(spend_low_tier)
            print(n_conversions_high_tier)
            print(spend_high_tier)
            t_marginal_dict = calc_marginal_CAC(n_conversions_low_tier, spend_low_tier, n_conversions_high_tier, spend_high_tier)
            df_CAC.loc[t_channel, t_df_CAC_colnames] = [t_marginal_dict[x] for x in base_col_names]

    CAC_dfs.update({model_type: df_CAC})

# print implied CAC
for m in CAC_dfs.keys():
    print('\n%s attribution model implied CAC:' % m)
    print(CAC_dfs[m][['marginal_CAC_t1', 'marginal_CAC_t2', 'marginal_CAC_t3']])

# write marginal CAC output
if write_to_file:
    for key, value in CAC_dfs.items():
        with open(key + '_model_marginal_implied_CAC.csv', 'w') as f:
            value.to_csv(f)

