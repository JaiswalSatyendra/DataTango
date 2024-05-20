from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from math import ceil
from threading import Thread
import pandas as pd
import logging
from logging.config import fileConfig
from collections import Counter
import re
import json
import os
import time
import numpy as np
import llm_helper
import boto3

fileConfig("logging_config.ini", disable_existing_loggers=False)
logger = logging.getLogger()


def nps(num_df, total_responses):
    # check min max value of nps
    recommend_df = num_df[
        (num_df["title"].str.lower().str.contains("recommend", case=False))
        | (num_df["type"] == "nps")
    ]["final_answer"]
    max_value = recommend_df.max()
    if max_value > 5:
        promoter_range = (9, 10)
        passive_range = (7, 8)
        negative_range = (1, 6)
    else:
        promoter_range = (4, 5)
        passive_range = (2.01, 3)
        negative_range = (1, 2)

    promoters = recommend_df[
        ((recommend_df >= promoter_range[0]) & (recommend_df <= promoter_range[1]))
    ].shape[0]

    neutral = recommend_df[
        ((recommend_df >= passive_range[0]) & (recommend_df <= passive_range[1]))
    ].shape[0]

    detractors = recommend_df[
        ((recommend_df >= negative_range[0]) & (recommend_df <= negative_range[1]))
    ].shape[0]
    # NPS score
    nps = round((promoters - detractors) * 100 / total_responses, 1)
    postive_satisfaction = round(promoters * 100 / total_responses, 1)
    neutral_satisfaction = round(
        neutral * 100 / total_responses,
        1,
    )
    negative_satisfaction = round(detractors * 100 / total_responses, 1)

    nps_dict = {
        "positive": postive_satisfaction,
        "negative": negative_satisfaction,
        "neutral": neutral_satisfaction,
    }
    extra_values = round(
        100
        - (
            round(nps_dict["neutral"], 1)
            + round(nps_dict["positive"], 1)
            + round(nps_dict["negative"], 1)
        ),
        1,
    )
    # print(extra_values)
    nps_dict_formatted = {
        "neutral": round(nps_dict["neutral"], 1),
        "positive": round(nps_dict["positive"], 1),
        "negative": (round(nps_dict["negative"] + extra_values, 1)),
    }

    return nps, nps_dict_formatted, max_value


def map_nps_category(x, max_value):
    if max_value > 5:
        promoter_range = (9, 10)
        passive_range = (7, 8)
        negative_range = (1, 6)
    else:
        promoter_range = (4, 5)
        passive_range = (2.01, 3)
        negative_range = (1, 2)
    if x >= promoter_range[0] and x <= promoter_range[1]:
        return "Promoter"
    elif x >= passive_range[0] and x <= passive_range[1]:
        return "Passive"
    elif x >= negative_range[0] and x <= negative_range[1]:
        return "Detractor"
    else:
        # TODO remove this.
        return "Promoter"


def satisfaction_breakdown(contact_df, total_responses):
    # satisfaction breakdown

    postive_satisfaction = round(
        (contact_df[contact_df["mean"] >= 4].shape[0]) * 100 / total_responses,
        1,
    )
    neutral_satisfaction = round(
        (contact_df[(contact_df["mean"] >= 3) & (contact_df["mean"] < 4)].shape[0])
        * 100
        / total_responses,
        1,
    )
    negative_satisfaction = round(
        (contact_df[contact_df["mean"] < 3].shape[0]) * 100 / total_responses, 1
    )
    satisfaction_dict = {
        "positive": postive_satisfaction,
        "negative": negative_satisfaction,
        "neutral": neutral_satisfaction,
    }
    extra_values = round(
        100
        - (
            round(satisfaction_dict["neutral"], 1)
            + round(satisfaction_dict["positive"], 1)
            + round(satisfaction_dict["negative"], 1)
        ),
        1,
    )
    # print(extra_values)
    satisfaction_dict_formatted = {
        "neutral": round(satisfaction_dict["neutral"], 1),
        "positive": round(satisfaction_dict["positive"], 1),
        "negative": (round(satisfaction_dict["negative"] + extra_values, 1)),
    }
    return satisfaction_dict_formatted


def topic_sentiment(num_df, field_type="all"):
    keywords = ["age"]
    pattern = "|".join(keywords)
    num_df = num_df[~num_df["title"].str.lower().str.contains(pattern, case=False)]
    # Create a dictionary to map values to categories
    rating_map = {
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive",
    }

    if field_type != "all":
        num_df = num_df[num_df["type"] == field_type]
    # Apply the mapping to create a new column 'sentiment'
    num_df["sentiment"] = num_df["final_answer"].map(rating_map)
    # Group by 'category' and 'sentiment', then calculate the percentage
    df_cat_sum = (
        num_df.groupby(["category", "sentiment"])["sentiment"]
        .count()
        .unstack(fill_value=0)
    )
    df_cat_sum = (df_cat_sum.divide(df_cat_sum.sum(axis=1), axis=0) * 100).reset_index()
    # Add missing columns if not present
    missing_columns = set(["neutral", "negative", "positive"]) - set(df_cat_sum.columns)
    for col in missing_columns:
        df_cat_sum[col] = 0
    cat_sum_dict = df_cat_sum.to_dict("index")

    formated_cat_sum = {}
    for values in cat_sum_dict.values():
        topic = values["category"]
        extra_values = round(
            100
            - (
                round(values["neutral"], 1)
                + round(values["positive"], 1)
                + round(values["negative"], 1)
            ),
            1,
        )
        # print(extra_values)
        formated_cat_sum[topic] = {
            "neutral": round(values["neutral"], 1),
            "positive": round(values["positive"], 1),
            "negative": (round(values["negative"] + extra_values, 1)),
        }

    return formated_cat_sum


def company_consideration(num_df):
    cc_kpi = round(round(num_df.final_answer.mean(), 2) * 2 * 10, 1)
    df_cc_grouped = (
        num_df.groupby("category")["final_answer"].mean() * 2
    ).reset_index()
    df_cc_grouped["category"] = df_cc_grouped["category"].str.replace("_", " ")
    df_cc_grouped["final_answer"] = df_cc_grouped["final_answer"].astype(float)
    df_cc_grouped["final_answer"] = df_cc_grouped["final_answer"].round(1)
    cc_dict = df_cc_grouped.set_index("category")["final_answer"].to_dict()
    cc_dict["overall"] = cc_kpi
    return cc_dict


def pivot_df(num_df):
    contact_df = num_df.pivot_table(
        index="response_id", columns="category", values="final_answer"
    ).reset_index()
    # contact_df["name"] = pd.merge(
    #     contact_df,
    #     num_df[num_df["category"].isin(["NAME"])][["response_id", "value"]],
    #     on="response_id",
    # )["value"]
    contact_df["mean"] = contact_df[
        contact_df.columns.difference(["response_id"])
    ].mean(axis=1)
    # contact_df["sat_mean"] = contact_df[
    #     [col for col in satisfaction_list if col in contact_df.columns]
    # ].mean(axis=1)
    return contact_df


def churn_risk(contact_df, total_responses):
    low_churn = (contact_df[contact_df["mean"] >= 3].shape[0]) / total_responses
    med_churn = (
        contact_df[(contact_df["mean"] >= 2) & (contact_df["mean"] < 3)].shape[0]
    ) / total_responses
    high_churn = (contact_df[contact_df["mean"] < 2].shape[0]) / total_responses
    overall_churn = (contact_df[contact_df["mean"] < 3].shape[0]) / total_responses
    churn_dict = {
        "overall_churn": round(overall_churn * 100),
        "low_churn": round(low_churn * 100, 1),
        "med_churn": round(med_churn * 100, 1),
        "high_churn": round(high_churn * 100, 1),
    }
    return churn_dict


# Define a function to calculate mean and return label
def categorize_churn(value):
    if value < 3 and value >= 2:
        return "medium"
    elif value >= 3:
        return "low"
    elif value < 2:
        return "high"


def categorize_sat(row):
    value = row["mean"]
    if value >= 3 and value < 4:
        return pd.Series(["neutral", row["neutral_categories"]])
    elif value >= 4:
        return pd.Series(["positive", row["positive_categories"]])
    elif value < 3:
        return pd.Series(["negative", row["negative_categories"]])


def get_prepped_df(contact_df):
    category_cols = [
        col
        for col in contact_df.columns
        if col
        not in [
            "nps_category",
            "name",
            "gender",
            "email",
            "hidden_email",
            "positive_categories",
            "neutral_categories",
            "negative_categories",
            "churn_risk",
            "sat_category",
            "topics",
            "age",
            "response_id",
            "mean",
        ]
    ]
    pos_cols = contact_df[category_cols].apply(
        lambda row: list(row[row >= 4].index), axis=1
    )
    neu_cols = contact_df[category_cols].apply(
        lambda row: list(row[row == 3].index), axis=1
    )
    neg_cols = contact_df[category_cols].apply(
        lambda row: list(row[row < 3].index), axis=1
    )

    contact_df["positive_categories"] = (
        pos_cols.astype(str).str.replace("[", "").str.replace("]", "")
    )
    contact_df["neutral_categories"] = (
        neu_cols.astype(str).str.replace("[", "").str.replace("]", "")
    )
    contact_df["negative_categories"] = (
        neg_cols.astype(str).str.replace("[", "").str.replace("]", "")
    )

    # contact_df["churn_risk"] = contact_df["mean"].apply(categorize_churn)
    contact_df[["sat_category", "topics"]] = contact_df.apply(categorize_sat, axis=1)
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    labels = [
        "10-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81-90",
    ]
    if "age" in contact_df.columns:
        if contact_df["age"].str.contains("-").any():
            contact_df["age_bucket"] = contact_df["age"]
        else:
            contact_df["age_bucket"] = pd.cut(
                pd.to_numeric(contact_df["age"]), bins=bins, labels=labels
            )

    return contact_df


def pie_chart_data(cat_df):
    cat_df_no_feedback = cat_df[cat_df["type"] != "string"]
    grouped_cat_df = (
        cat_df_no_feedback.groupby(["category", "final_answer"])
        .size()
        .reset_index(name="count")
    )

    # Filter for titles with 10 or less unique final answers
    filtered_df = grouped_cat_df.groupby("category").filter(
        lambda x: len(x["final_answer"].unique()) <= 10
    )

    # Convert the grouped DataFrame to the desired dictionary format.
    cat_count_dict = {}
    for title, group in filtered_df.groupby("category"):
        cat_count_dict[title] = {
            "series": group["count"].tolist(),
            "labels": group["final_answer"].tolist(),
        }

    return cat_count_dict


def box_plot(contact_df, cat_df):
    contact_temp = contact_df.copy()
    grouped_cat_df = (
        cat_df.groupby(["title", "final_answer"]).size().reset_index(name="count")
    )

    # Filter for titles with 10 or less unique final answers
    filtered_df = grouped_cat_df.groupby("title").filter(
        lambda x: len(x["final_answer"].unique()) <= 10
    )

    box_plot_categories = filtered_df.title.unique()
    for i in box_plot_categories:
        # check if related title is present in cat_df
        temp_df = cat_df[cat_df["title"] == i]
        if not temp_df.empty:
            contact_temp[i] = contact_temp.merge(temp_df, on="response_id", how="left")[
                "answer"
            ]

    contact_temp["mean"] = contact_temp["mean"].round(1)
    chart_data = {}

    # Loop through each category of interest
    for category in box_plot_categories:
        pass
        # Initialize data schema for current category
        chart_data[category] = {"data": []}

        # Group by the category and calculate descriptive stats
        grouped_df = contact_temp.groupby(category)["mean"].describe(
            percentiles=[0.25, 0.5, 0.75]
        )

        for value, stats in grouped_df.iterrows():
            # Append x and y data points into structured format
            chart_data[category]["data"].append(
                {
                    "x": value,
                    "y": [
                        stats["min"],
                        stats["25%"],
                        stats["50%"],
                        stats["75%"],
                        stats["max"],
                    ],
                }
            )
    return chart_data


def ts_word_cloud(contact_df, cat_df, model="gpt-3.5-turbo-1106"):
    filter_ques = ["email", "name", "location", "address", "city", "zip"]
    filtered_text = cat_df[
        (cat_df["type"].isin(["short_text", "long_text", "string"]))
        & (~cat_df["title"].str.contains("|".join(filter_ques, case=False, regex=True)))
    ]
    if ("short_text" in filtered_text["type"].unique()) or (
        "long_text" in filtered_text["type"].unique()
        or ("string" in filtered_text["type"].unique())
    ):
        filtered_text_pivoted = filtered_text.pivot_table(
            index=["response_id"],
            columns="category",
            values="final_answer",
            aggfunc="first",
        ).reset_index()
        text_categories = [
            i for i in filtered_text_pivoted.columns if i != "response_id"
        ]
        # Step 1: Merge the dataframes on response_id

        temp_merged_df = pd.merge(contact_df, filtered_text_pivoted, on="response_id")

        # get index, text_categories
        # group the sentences in 5 sentences and pass to open ai
        # pass them parallely and save the results in a json file
        # do the above two steps in parallel for each text category
        # collate all the results and count each topic's frequency, sentiment and response_id
        # iterate over each row of the dataframe

        parent_id = str(int(time.time())) + "_" + model
        final_path = os.path.join("/tmp", parent_id)

        if not os.path.exists(final_path):
            os.makedirs(final_path)

        for col in text_categories:
            output_path = classify_questions(temp_merged_df, col, model, final_path)

        output_payload, prompt_tokens, completion_tokens = (
            aggregate_classification_results(output_path, temp_merged_df)
        )

        # check with df based on id get response_id
        logger.info(f"prompt_tokens: {prompt_tokens}")
        logger.info(f"completion_tokens: {completion_tokens}")
        return output_payload

    else:
        logger.info("No free text based data")
        return {}


def create_query(contact_df):
    athena_list = [
        "name",
        "nps_category",
        "sat_category",
        "average_purchase_value",
        "purchase_frequency",
        "cart_drop_rate",
        "topics",
        "age",
        "gender",
        "churn_risk",
        "PC1",
        "PC2",
        "mean",
        "cluster",
        "email",
    ]
    query_list = []
    for i in athena_list:
        if i in contact_df.columns:
            query_list.append(i)
    return query_list


def auto_scaler(num_df):
    def custom_scale(s):
        return (s / s.max()) * 5

    num_df["final_answer"] = num_df.groupby("title")["final_answer"].transform(
        custom_scale
    )

    return num_df


def classify_questions(df, col, model_name, final_path):
    """
    Creates threads to classify the questions using an LLM.
    """

    def split_dataframe(df, col, chunk_size):
        chunks = []
        num_chunks = (
            len(df) // chunk_size
            if len(df) % chunk_size == 0
            else len(df) // chunk_size + 1
        )

        for i in range(num_chunks):
            chunk_dict = df[i * chunk_size : (i + 1) * chunk_size][col].to_dict()
            if chunk_dict:  # This will ensure no empty dictionary is added
                chunks.append(chunk_dict)
        return chunks

    chunks = split_dataframe(df, col, 20)

    classifer_thread_list = []
    # parent_id = str(int(time.time())) + "_" + model_name
    # output_path = os.path.join(gparent_path, parent_id)

    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    for comments in chunks:
        thread_object = Thread(
            target=llm_helper.main,
            args=(
                "topic-sentiment",
                comments,
                final_path,
                0,
                model_name,
            ),
        )
        classifer_thread_list.append(thread_object)
        thread_object.start()

    for thread in classifer_thread_list:
        thread.join()

    return final_path


def aggregate_classification_results(output_path, df):
    """
    Aggregates the classification results from individual files into a collective payload.
    """
    output_dict = {
        "Detractor": {"data": [], "labels": [], "response_id": [], "color": []},
        "Promoter": {"data": [], "labels": [], "response_id": [], "color": []},
        "Passive": {"data": [], "labels": [], "response_id": [], "color": []},
        "Overall": {"data": [], "labels": [], "response_id": [], "color": []},
    }

    topic_counter = {
        "Detractor": {
            "negative": {},
            "very negative": {},
            "very positive": {},
            "neutral": {},
            "positive": {},
        },
        "Promoter": {
            "negative": {},
            "very negative": {},
            "very positive": {},
            "neutral": {},
            "positive": {},
        },
        "Passive": {
            "negative": {},
            "very negative": {},
            "very positive": {},
            "neutral": {},
            "positive": {},
        },
        "Overall": {
            "negative": {},
            "very negative": {},
            "very positive": {},
            "neutral": {},
            "positive": {},
        },
    }
    file_list = os.listdir(output_path)
    logger.info(f"file_list = {file_list}")
    output_payload = {}
    payload_list, prompt_tokens, completion_tokens = [], 0, 0
    stop_words = open("stopwords.json").read()
    stop_words_list = json.loads(stop_words)["stopwords"]
    for file in file_list:
        file_object = open(f"{output_path}/{file}", "r").read()
        file_payload = json.loads(file_object)
        try:
            if type(file_payload["comments"]) == dict:
                for key, value in file_payload["comments"].items():
                    index = int(key)
                    response_id = df.loc[index]["response_id"]
                    nps_cat = df.loc[index]["nps_category"]
                    if type(value) == dict:
                        for topic, sentiment in value.items():
                            if (
                                (topic != "null")
                                and (topic != "sentiment")
                                and (sentiment != "null")
                                and (len(topic) > 2)
                                and (topic not in stop_words_list)
                            ):
                                if topic in topic_counter[nps_cat][sentiment].keys():
                                    topic_counter[nps_cat][sentiment][topic][
                                        "count"
                                    ] += 1
                                    topic_counter[nps_cat][sentiment][topic][
                                        "response_id"
                                    ].append(response_id)
                                else:
                                    topic_counter[nps_cat][sentiment][topic] = {}
                                    topic_counter[nps_cat][sentiment][topic][
                                        "count"
                                    ] = 1
                                    topic_counter[nps_cat][sentiment][topic][
                                        "response_id"
                                    ] = [response_id]
                                if topic in topic_counter["Overall"][sentiment].keys():
                                    topic_counter["Overall"][sentiment][topic][
                                        "count"
                                    ] += 1
                                    topic_counter["Overall"][sentiment][topic][
                                        "response_id"
                                    ].append(response_id)
                                else:
                                    topic_counter["Overall"][sentiment][topic] = {}
                                    topic_counter["Overall"][sentiment][topic][
                                        "count"
                                    ] = 1
                                    topic_counter["Overall"][sentiment][topic][
                                        "response_id"
                                    ] = [response_id]
        except Exception as e:
            logger.info(e)
            logger.info(file)
            logger.info(file_payload)

        prompt_tokens += file_payload["usage"]["prompt_tokens"]
        completion_tokens += file_payload["usage"]["completion_tokens"]

    color_mapping = {
        "very positive": "#0ab054",
        "positive": "#1bd26d",
        "neutral": "#feac00",
        "negative": "#fe6779",
        "very negative": "#f1334b",
    }

    # logger.info(topic_counter)
    # logger.info("topic_counter created")
    for nps_cat, value in topic_counter.items():
        data_list = []
        labels_list = []
        color_list = []
        response_id_list = []
        for sentiment, topic_info in value.items():
            if topic_info != {}:
                for topic, count_info in topic_info.items():
                    labels_list.append(topic)
                    data_list.append(count_info["count"])
                    response_id_list.append(count_info["response_id"])
                    color_list.append(color_mapping[sentiment])

        output_dict[nps_cat]["data"] = data_list
        output_dict[nps_cat]["labels"] = labels_list
        output_dict[nps_cat]["response_id"] = response_id_list
        output_dict[nps_cat]["color"] = color_list

    output_payload["word_cloud"] = output_dict

    return output_payload, prompt_tokens, completion_tokens


def get_silver_data(s3_key, env):
    start_time = time.time()
    s3 = boto3.client("s3")
    bucket_name = f"convertml-{env}-data"
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    df = pd.read_csv(obj["Body"])
    end_time = time.time()
    logger.info(f"download time {end_time - start_time}")
    return df


def create_crosstab(num_df, cat_df, ques=[], values=[]):

    merged_df = pd.concat([num_df, cat_df])
    df_meta = merged_df[
        ["id_x", "title", "type", "ans_type", "options"]
    ].drop_duplicates()
    pivoted_df = merged_df.pivot_table(
        values="answer",
        index=["response_id", "submitted_at"],
        columns=["title"],
        aggfunc="first",
    ).reset_index()

    duplicate_counts = merged_df.groupby(["response_id", "submitted_at"]).size()
    duplicate_counts = duplicate_counts[
        duplicate_counts > 1
    ]  # Filter groups with more than 1 occurrence
    logger.info(duplicate_counts)

    unique_counts = pivoted_df.nunique()

    # Filter columns with more than 10 distinct values
    filtered_columns = unique_counts[unique_counts <= 10].index

    # Select the filtered columns from the original dataframe
    filtered_df = pivoted_df[filtered_columns]
    filtered_df = filtered_df[
        filtered_df.columns[~filtered_df.columns.isin(["response_id", "submitted_at"])]
    ]
    if (len(ques) != 0) and (len(values) != 0):
        # temp_df = filtered_df.copy()
        # for ques, value in zip(ques, values):
        #     temp_df = temp_df[temp_df[ques] == value]
        overall_condition = pd.Series(
            [False] * len(filtered_df), index=filtered_df.index
        )

        # Loop through each question/value pair and update the overall condition accordingly
        for ques, value in zip(ques, values):
            condition = filtered_df[ques] == value
            overall_condition = overall_condition | condition

        # Use the overall_condition to filter your DataFrame
        temp_df = filtered_df[overall_condition]
    else:
        temp_df = filtered_df.copy()

    result = []
    # Iterate over the columns
    for col in temp_df.columns:
        # Calculate value count % of distinct options in each column
        value_counts = (
            temp_df[col]
            .value_counts(dropna=False)
            .reindex(pivoted_df[col].unique(), fill_value=0)
        )

        # Replace the NaN index with 'Not Answered' if it exists
        if pd.isna(value_counts.index).any():
            new_index = value_counts.index.fillna("Not Answered")
            value_counts.index = new_index

        # Calculate percentage and format as string
        value_counts_percent = (value_counts / len(temp_df) * 100).round(2).astype(
            str
        ) + "%"

        # Create a dictionary for the column
        col_dict = {col: value_counts_percent.to_dict()}

        # Add the dictionary to the result list
        result.append(col_dict)

    return result


def get_ml_stats(final_df, num_cols, cat_cols):
    # clustering analysis with both data
    # Function to get the first mode or a default value if empty (no mode found)
    def get_first_mode(series):
        modes = series.mean()
        return modes

    # Group by 'Cluster' and then aggregate using the custom function on text_columns

    num_cols_list = [col for col in num_cols if (col in final_df.columns)]
    final_df[num_cols_list] = final_df[num_cols_list].apply(pd.to_numeric)
    common_trends = final_df.groupby("cluster")[num_cols_list].agg(get_first_mode)
    return common_trends


def save_table_parallel_s3(bucket_name, s3_key, merged_df):

    s3_client = boto3.client("s3")
    multipart_upload = s3_client.create_multipart_upload(Bucket=bucket_name, Key=s3_key)
    part_info = {"Parts": []}

    part_size = 10 * 1024 * 1024  # 10 MB

    # Convert DataFrame to CSV and encode to bytes directly in a streaming fashion
    with io.StringIO() as csv_buffer:
        merged_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue().encode()

        total_size = len(csv_content)
        part_count = ceil(total_size / part_size)

        def upload_part(part):
            start_range = part * part_size
            end_range = min((part + 1) * part_size, total_size)
            part_data = csv_content[start_range:end_range]

            part_response = s3_client.upload_part(
                Bucket=bucket_name,
                Key=s3_key,
                PartNumber=part + 1,
                UploadId=multipart_upload["UploadId"],
                Body=part_data,
            )

            logger.info(f"Part {part + 1}/{part_count} uploaded successfully")
            return {"PartNumber": part + 1, "ETag": part_response["ETag"]}

        # Use ThreadPoolExecutor to upload parts in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_part = {
                executor.submit(upload_part, part): part for part in range(part_count)
            }
            for future in as_completed(future_to_part):
                part_info["Parts"].append(future.result())
                uploaded_parts = len(part_info["Parts"])
                progress = (uploaded_parts / part_count) * 100
                # self.progress_updater(progress)
                logger.info(
                    f"Progress: {uploaded_parts} of {part_count} parts uploaded ({progress:.2f}%)."
                )

        # Clean up the large in-memory object explicitly
        del csv_content

    # Sort parts by PartNumber before completing the multipart upload
    part_info["Parts"].sort(key=lambda x: x["PartNumber"])

    # Complete multipart upload
    s3_client.complete_multipart_upload(
        Bucket=bucket_name,
        Key=s3_key,
        UploadId=multipart_upload["UploadId"],
        MultipartUpload=part_info,
    )

    logger.info("Upload complete.")
