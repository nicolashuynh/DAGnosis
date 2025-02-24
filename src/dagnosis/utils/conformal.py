# third party
import pandas as pd


def analyse_conformal_dict(conformal_dict: dict) -> pd.DataFrame:
    """
    Analyse the conformal dictionary and return a dataframe with the outliers and inconsistencies.
    """
    dictionnary_df = {}
    for feature in conformal_dict.keys():
        df = conformal_dict[feature]

        def func(truth, min_val, max_val, interval):

            return not ((truth >= min_val) & (truth <= max_val))

        df["outlier"] = df.apply(
            lambda x: func(
                x["true_val"],
                x["min"],
                x["max"],
                x["conf_interval"],
            ),
            axis=1,
        )

        dictionnary_df[feature] = df["outlier"].tolist()

    final_df = pd.DataFrame(dictionnary_df)
    isInconsistent = final_df.any(axis=1)
    final_df["inconsistent"] = isInconsistent
    return final_df
