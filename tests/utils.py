def pd_dataframe2bq_query(df):
    query = ""

    df_edit = df.applymap(lambda s: f'"{s}"' if type(s) == str else s).fillna('NULL')

    for index in df_edit.index:
        query += "SELECT "
        for column in df.columns:
            query += f"{df_edit.loc[index, column]} AS {column}"
            if column != df_edit.columns[-1]:
                query += ", "
            elif index != df_edit.index[-1]:
                query += " UNION ALL\n"

    return query