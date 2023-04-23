
import pandas as pd
import numpy as np
from datetime import datetime

import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
params = {
    "figure.figsize":(10,6),
    "text.color":"#162871",
    "axes.titlesize":16,
    "axes.labelsize":14,
    "axes.labelcolor": "#162871",
    "axes.edgecolor": "#162871",
    "xtick.color": "#162871",
    "ytick.color": "#162871",
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "legend.fontsize":12, 
    "axes.grid.axis":"y",
    "axes.spines.left":False,
    "axes.spines.top":False,
    "axes.spines.right":False,

}

plt.rcParams.update(params)

class clv_cohort:
    
#     def __init__(self):
#         pass

    def get_data(self, data_df=None):

        url = "../dataBase/Transactions.csv"

        df_raw = pd.read_csv(url)

        self.df = (
            df_raw
            .assign(transaction_date = lambda X: pd.to_datetime(X["transaction_date"]))
        )
        grouped = self.df.groupby(["customer_id"])

        self.df["min_year"] = grouped["transaction_date"].transform(lambda x: x.min().year)
        self.df["Year"] = grouped["transaction_date"].transform(lambda x: x.dt.year)
        self.df["max_year"] = grouped["transaction_date"].transform(lambda x: x.max().year)
        if data_df:
            return self.df 

    def get_transaction_trends_data(self):
        if not hasattr(self, "df"):
            self.get_data()

        self.df_dly = (
            self.df
            .assign(year_month = lambda X: X["transaction_date"].apply(lambda x: x.strftime("%Y-%m")))
            .groupby("year_month", as_index=False)
            .agg(
                orders = ("customer_id", "count"),
                customers = ("customer_id", "nunique"),
            )
            .sort_values("year_month")
            .assign(order_pct_change = lambda X: X["orders"].pct_change())
            .assign(dropped = lambda X: X["order_pct_change"].apply(lambda x: "red" if x<0 else 'green' ) )
        )

    def get_aquisition_data(self):
        if not hasattr(self, "df"):
            self.get_data()

        self.df_acquisition = (
            self.df
            .groupby("min_year", as_index=False)
            .agg(customers = ("customer_id", "nunique"))
            .sort_values("min_year")
            .assign(cust_pct_change = lambda X: X["customers"].pct_change())
        )

    def get_clv_data(self):
        if not hasattr(self, "df"):
            self.get_data()

        if not hasattr(self, "df_acquisition"):
            self.get_aquisition_data()

        self.df_clv = (
            self.df
            .groupby(["min_year", "Year"], as_index=False)
            .agg(
                customers_trxn = ("customer_id", "nunique"),
                trxns = ("transaction_id", "count"),
                amount = ("amount", "sum"),
            )
            .reset_index(drop=True)
            .merge(self.df_acquisition[["min_year", "customers"]], how='left', on='min_year',)
            .assign(
                avg_trxn = lambda X: X["trxns"]/X["customers_trxn"],
                avg_amount = lambda X: X["amount"]/X["customers_trxn"],
                retention = lambda X: X["customers_trxn"]/X["customers"],
            )
            .assign(
                Cumm_amount = lambda X: X.groupby("min_year")["amount"].transform(lambda x: x.cumsum()),
                annual_clv = lambda X: X["Cumm_amount"]/X["customers"],
                months = lambda X: ((X["Year"] - X["min_year"]) + 1) * 12,
            )
            
        )


    def get_weighted_clv_data(self):
        if not hasattr(self, "df_clv"):
            self.get_clv_data()

        self.df_wclv = (
            self.df_clv
            .assign(weighted_cohort_clv = lambda X: X["annual_clv"] * X["customers"])
            .groupby("months", as_index=False)
            .agg(
                amount = ("weighted_cohort_clv", "sum"),
                customers = ("customers", "sum"),
            )
            .assign(weighted_clv = lambda X: X["amount"]/X["customers"])
        )

    def get_prediction_data(self):
        if not hasattr(self, "ols_model"):
            self.fit_ols_model()

        self.df_clv_pred = (
            pd.DataFrame()
            .assign(
                months = np.arange(1, 109),
                predicted_clv = lambda X: self.ols_model.predict(X["months"])
            )
        )


    def get_customer_plot(self):
        if not hasattr(self, "df_acquisition"):
            self.get_aquisition_data()

        fig = px.line(
            self.df_acquisition, 
            x="min_year", 
            y="customers", 
            markers=True, 
            line_shape="spline", 
            color_discrete_sequence=["deepskyblue"],
        )

        fig.add_trace(
            go.Scatter(
                x=self.df_acquisition["min_year"], 
                y=self.df_acquisition["cust_pct_change"], 
                mode='lines+markers',
                line_shape='spline',
                name="MoM", 
                yaxis="y2",
                line=dict(color="#e90076", width=1),
            )
        )
                                 

        fig.update_layout(
            
            plot_bgcolor="white",

            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside"),

            yaxis=dict(showline=False, showgrid=True, gridcolor="lightgray"),
            
            yaxis2=dict(title="Percentage change", overlaying="y", side="right", showgrid=False, tickformat=".0%"),

            legend=dict(orientation="h", x=0.5, y=1.1),

            font=dict(size=12, color="#5b68f6"),

            title=dict(font=dict(size=20, color="#5b68f6") ,x=0.5, xanchor="center"),
        )

        return fig

    def get_transaction_plot(self):
        if not hasattr(self, "df_dly"):
            self.get_transaction_trends_data()

        fig = px.line(
            self.df_dly, 
            x="year_month", 
            y="orders", 
            markers=True, 
            line_shape="spline", 
            color_discrete_sequence=["deepskyblue"],
        )
                      

        fig.add_trace(
            go.Scatter(
                x=self.df_dly["year_month"], 
                y=self.df_dly["order_pct_change"], 
                mode='lines+markers',
                line_shape='spline',
                name="MoM", 
                yaxis="y2",
                line=dict(color="#e90076", width=1),
            )
        )
                                 

        fig.update_layout(
            plot_bgcolor="white",

            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside"),

            yaxis=dict(showline=False, showgrid=True, gridcolor="lightgray"),
            
            yaxis2=dict(title="Percentage change", overlaying="y", side="right", showgrid=False, tickformat=".0%"),

            legend=dict(orientation="h", x=0.5, y=1.1),

            font=dict(size=12, color="#5b68f6"),

            title=dict(font=dict(size=20, color="#5b68f6") ,x=0.5, xanchor="center"),
        )

        return fig

    def get_retention_plot(self):
        if not hasattr(self, "df_clv"):
            self.get_clv_data()

        fig = px.line(
            self.df_clv, 
            x="Year", 
            y="retention", 
            color="min_year",
            markers=True, 
            line_shape="spline", 
        )

        fig.update_layout(
            
            plot_bgcolor="white",

            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside"),

            yaxis=dict(showline=False, showgrid=True, gridcolor="lightgray", tickformat=".0%"),

            legend=dict(title="cohort", orientation="h", x=0.5, y=1.1),

            font=dict(size=12, color="#5b68f6"),

            title=dict(font=dict(size=20, color="#5b68f6") ,x=0.5, xanchor="center"),
        )

        return fig

    def get_clv_heatmaps(self, figsize=(32,5)):
        if not hasattr(self, "df_clv"):
            self.get_clv_data()

        columns = ["avg_trxn", "avg_amount"]

        colors = ["flare", "crest"]

        fig, ax = plt.subplots(1,2, figsize=figsize)

        for i in range(2):
            df_avg_plot = (
                self.df_clv
                .pivot(index='min_year', columns='Year', values=columns[i])
                .sort_index(ascending=False)
            )
            sns.heatmap(df_avg_plot, annot=True, cmap=colors[i], cbar=True, ax=ax[i])
        plt.subplots_adjust(wspace=0.0)
        return fig

    def get_annual_clv_plot(self):
        if not hasattr(self, "df_clv"):
            self.get_clv_data()

        fig = px.line(
            self.df_clv, 
            x="Year", 
            y="annual_clv", 
            color="min_year",
            markers=True, 
            line_shape="spline", 
        )

        fig.update_layout(
            
            width=1000,
            
            plot_bgcolor="white",

            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside"),

            yaxis=dict(showline=False, showgrid=True, gridcolor="lightgray"),

            legend=dict(title="cohort", orientation="h", x=0.4, y=1.1),

            font=dict(size=12, color="#5b68f6"),

            title=dict(font=dict(size=20, color="#5b68f6") ,x=0.5, xanchor="center"),
        )

        return fig

    def get_monthly_clv_plot(self):
        if not hasattr(self, "df_clv"):
            self.get_clv_data()

        fig = px.line(
            self.df_clv, 
            x="months", 
            y="annual_clv", 
            color="min_year",
            markers=True, 
            line_shape="spline", 
        )
        fig.update_layout(
            
            width=1000,
            
            plot_bgcolor="white",

            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside"),

            yaxis=dict(showline=False, showgrid=True, gridcolor="lightgray"),

            legend=dict(title="cohort", orientation="h", x=0.4, y=1.1),

            font=dict(size=12, color="#5b68f6"),

            title=dict(font=dict(size=20, color="#5b68f6") ,x=0.5, xanchor="center"),
        )
        return fig

    def get_weighted_clv_plot(self):
        if not hasattr(self, "df_wclv"):
            self.get_weighted_clv_data()

        if not hasattr(self, "df_clv"):
            self.get_clv_data()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.df_wclv["months"], 
                y=self.df_wclv["weighted_clv"], 
                line_shape='spline',
                name="Weighted CLV", 
                line=dict(color="#e90076", width=3),
            )
        )

        cohorts={2010:'red', 2011:'steelblue', 2012:'deepskyblue', 2013:'teal', 2014:'turquoise', 2015:'gold'}

        for cohort in cohorts:
            df_plot = self.df_clv.query("min_year==@cohort")
            fig.add_trace(
                go.Scatter(
                    x=df_plot["months"], 
                    y=df_plot["annual_clv"], 
                    mode='markers',
                    marker=dict(color=cohorts.get(cohort), size=10),
                    name=cohort,
                )
            )

        fig.update_layout(
            
            width=1000,
            
            plot_bgcolor="white",

            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside"),

            yaxis=dict(showline=False, showgrid=True, gridcolor="lightgray"),

            legend=dict(title="cohort", orientation="h", x=0.25, y=1.1),

            font=dict(size=12, color="#5b68f6"),

            title=dict(font=dict(size=20, color="#5b68f6") ,x=0.5, xanchor="center"),
        )

        return fig


    def fit_ols_model(self):
        if not hasattr(self, "df_wclv"):
            self.get_weighted_clv_data()

        self.ols_model = smf.ols('weighted_clv ~ months', data=self.df_wclv).fit()

        return self.ols_model

    def get_clv_prediction_plot(self):

        if not hasattr(self, "df_wclv"):
            self.get_weighted_clv_data()

        if not hasattr(self, "df_clv_pred"):
            self.get_prediction_data()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.df_wclv["months"], 
                y=self.df_wclv["weighted_clv"], 
                line_shape='spline',
                name="Weighted CLV", 
                line=dict(color="#e90076", width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.df_clv_pred["months"], 
                y=self.df_clv_pred["predicted_clv"], 
                line_shape='spline',
                name="Predicted CLV", 
                mode="lines+markers", 
                marker=dict(size=5),
                line=dict(color="green", width=1),
            )
        )

        cohorts={2010:'red', 2011:'steelblue', 2012:'deepskyblue', 2013:'teal', 2014:'turquoise', 2015:'gold'}

        for cohort in cohorts:
            df_plot = self.df_clv.query("min_year==@cohort")
            fig.add_trace(
                go.Scatter(
                    x=df_plot["months"], 
                    y=df_plot["annual_clv"], 
                    mode='markers',
                    marker=dict(color=cohorts.get(cohort), size=10),
                    name=cohort,
                )
            )

        fig.update_layout(
            
            width=1000,
            
            plot_bgcolor="white",

            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside", tickmode="array", tickvals=list(range(0,109,6))),

            yaxis=dict(showline=False, showgrid=True, gridcolor="lightgray"),

            legend=dict(title="cohort", orientation="h", x=0.1, y=1.1),

            font=dict(size=12, color="#5b68f6"),

            title=dict(font=dict(size=20, color="#5b68f6") ,x=0.5, xanchor="center"),
        )

        return fig

