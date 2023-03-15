import numpy as np
import json
import pandas as pd
import pyrssa as prs
import os
from itertools import chain
from typing import Literal
import unittest

DATAFRAME_NAMES = ("co2", "fr1k.nz", "fr1k", "fr50.nz", "fr50")
DATAFRAME_KINDS = ("reconstruction", "rforecast.orig", "rforecast.rec", "vforecast")
KINDS_TRANSLATION = {"reconstruct": "reconstruction", "rforecast": "rforecast.orig", "vforecast": "vforecast"}
MAIN_DIRECTORY = os.path.join(os.getcwd(), "tests")
TEST_DATAFRAME_DIRECTORY = os.path.join(MAIN_DIRECTORY, "test_data")


def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def load_df(df_name, kinds: list, file_format="csv"):
    filename = os.path.join(TEST_DATAFRAME_DIRECTORY, df_name)
    print(filename)
    if kinds is None:
        kinds = os.listdir(filename)
    data_folders = [[os.path.join(filename, name), name]
                    for name in kinds
                    if os.path.isdir(os.path.join(filename, name))]

    df = {"series":
              pd.read_csv(os.path.join(filename, f"series.{file_format}")),
          "pars": None}

    params_path = os.path.join(filename, "pars.json")
    if os.path.exists(params_path):
        df["pars"] = read_json(params_path)

    for data_folder in data_folders:
        folder_files = [[os.path.join(data_folder[0], file), file] for file in os.listdir(data_folder[0])]
        df[data_folder[1]] = {file[1].replace(f".{file_format}", ""): pd.read_csv(file[0]) for file in folder_files}
    return df


def load_dataframes(df_names=DATAFRAME_NAMES, kinds=DATAFRAME_KINDS):
    return {df_name: load_df(df_name, kinds=kinds) for df_name in df_names}


def compute_reconstructions(x, Ls, groups,
                            kind: Literal["1d-ssa", "toeplitz-ssa"] = "1d-ssa",
                            svd_method: Literal["auto", "eigen", "propack", "nutrlan", "svd"] = "auto",
                            neig=None,
                            column_projector="none",
                            row_projector="none",
                            **kwargs):
    if neig is None:
        if isinstance(groups, dict):
            neig = np.max(list(chain(*groups.values())))
        elif isinstance(groups, list):
            neig = np.max([np.max(i) for i in groups])
        else:
            raise TypeError(f"Wrong type for groups: {type(groups)}")

    def compute_reconstruction(L):
        ss = prs.ssa(x, L, kind=kind, svd_method=svd_method,
                     neig=min(neig, L),
                     column_projector=column_projector,
                     row_projector=row_projector,
                     **kwargs)
        rec = prs.reconstruct(ss, groups=groups)
        return pd.DataFrame({key: rec[key] for key in rec.names})

    return {f"L{L_value}": compute_reconstruction(L_value) for L_value in Ls}


def compute_forecasts(x, Ls, groups, length,
                      kind: Literal["1d-ssa", "toeplitz-ssa"] = "1d-ssa",
                      forecast_method: Literal["recurrent", "vector"] = "recurrent",
                      base: Literal["reconstructed", "original"] = "reconstructed",
                      svd_method: Literal["auto", "eigen", "propack", "nutrlan", "svd"] = "auto",
                      neig=None,
                      column_projector="none",
                      row_projector="none",
                      **kwargs):
    if neig is None:
        if isinstance(groups, dict):
            neig = np.max(list(chain(*groups.values())))
        elif isinstance(groups, list):
            neig = np.max([np.max(i) for i in groups])
        else:
            raise TypeError(f"Wrong type for groups: {type(groups)}")

    def compute_forecast(L):
        ss = prs.ssa(x, L, kind=kind, svd_method=svd_method,
                     neig=min(neig, L),
                     column_projector=column_projector,
                     row_projector=row_projector,
                     **kwargs)
        if forecast_method == "recurrent":
            rec = prs.rforecast(ss, groups=groups, length=length, base=base, only_new=False)
        else:
            rec = prs.vforecast(ss, groups=groups, length=length, only_new=False)

        return pd.DataFrame({key: rec[key] for key in rec.names})

    return {f"L{L_value}": compute_forecast(L_value) for L_value in Ls}


def make_test_data(what: Literal["reconstruct", "rforecast", "vforecast"],
                   series,
                   name,
                   Ls,
                   groups,
                   Ls_forecast=None,
                   groups_forecast=None,
                   length=100,
                   kind: Literal["1d-ssa", "toeplitz-ssa"] = "1d-ssa",
                   svd_method: Literal["auto", "eigen", "propack", "nutrlan", "svd"] = "auto",
                   neig=None,
                   tolerance=1e-7,
                   svd_methods: Literal["auto", "eigen", "propack", "nutrlan", "svd"] = "auto",
                   svd_methods_forecast: Literal["auto", "eigen", "propack", "nutrlan", "svd"] = "auto",
                   column_projector="none",
                   row_projector="none"):
    result = {"series": series}

    if what == "reconstruct":
        result["reconstruction"] = compute_reconstructions(x=series, Ls=Ls, groups=groups, kind=kind,
                                                           svd_method=svd_method, neig=neig,
                                                           column_projector=column_projector,
                                                           row_projector=row_projector)

    return result


def test_test_data(what: Literal["reconstruct", "rforecast", "vforecast"],
                   test_data,
                   name=None,
                   Ls=None,
                   Ls_forecast=None,
                   svd_methods: list = None,
                   svd_methods_forecast: list = None,
                   neig=None,
                   tolerance=None,
                   column_projector="none",
                   row_projector="none",
                   kind=None,
                   **kwargs):
    if name is None:
        name = test_data["pars"]["name"]
    if Ls is None:
        Ls = test_data["pars"]["Ls"]["reconstruct"]
    if Ls_forecast is None:
        Ls_forecast = test_data["pars"]["Ls"].get("forecast", Ls)
    if svd_methods is None:
        svd_methods = test_data["pars"]["svd_methods"]["reconstruct"]
    if svd_methods_forecast is None:
        svd_methods_forecast = test_data["pars"].get("svd_methods_forecast", svd_methods)
    if kind is None:
        kind = test_data["pars"]["kind"]
    if tolerance is None:
        tolerance = test_data["pars"]["tolerance"]

    series = test_data["series"]
    groups = test_data["pars"]["groups"]["reconstruct"]
    forecast_groups = test_data["pars"]["groups"]["forecast"]
    length = test_data["pars"]["len"]

    print(f"Running {what} tests for {name}...")
    if what == "reconstruct":
        for i in range(len(Ls)):
            L = Ls[i]
            L_name = f"L{L}"
            for svd_method in svd_methods[i]:
                print(f"Data: {name}, Kind: {kind}, SVD-Method: {svd_method}, L: {L}...", end="")
                reconstruction = compute_reconstructions(series, [L], groups=groups, kind=kind, svd_method=svd_method,
                                                         neig=neig, column_projector=column_projector,
                                                         row_projector=row_projector)

                pd.testing.assert_frame_equal(reconstruction[L_name], test_data["reconstruction"][L_name],
                                              rtol=tolerance)
                print("OK")

    elif what == "rforecast":
        for i in range(len(Ls_forecast)):
            L = Ls_forecast[i]
            L_name = f"L{L}"
            for svd_method in svd_methods_forecast[i]:
                print(f"Data: {name}, Kind: {kind}, SVD-Method: {svd_method}, L: {L}...", end="")
                rforecast_orig = compute_forecasts(series, [L], groups=forecast_groups, length=length,
                                                   kind=kind, forecast_method="recurrent", base="original",
                                                   svd_method=svd_method, neig=neig,
                                                   column_projector=column_projector, row_projector=row_projector,
                                                   **kwargs)
                rforecast_rec = compute_forecasts(series, [L], groups=forecast_groups, length=length,
                                                  kind=kind, forecast_method="recurrent", base="reconstructed",
                                                  svd_method=svd_method, neig=neig,
                                                  column_projector=column_projector, row_projector=row_projector,
                                                  **kwargs)

                # if name == "co2" and kind == "1d-ssa" and svd_method == "svd" and L == 17:
                #     print(rforecast_orig[L_name])
                #     print(test_data["rforecast.orig"][L_name])
                #     print(rforecast_rec[L_name])
                #     print(test_data["rforecast.rec"][L_name])
                #     print(np.max(rforecast_orig[L_name] - test_data["rforecast.orig"][L_name]))

                pd.testing.assert_frame_equal(rforecast_orig[L_name], test_data["rforecast.orig"][L_name],
                                              rtol=tolerance)
                pd.testing.assert_frame_equal(rforecast_rec[L_name], test_data["rforecast.rec"][L_name],
                                              rtol=tolerance)
                print("OK")

    elif what == "vforecast":
        for i in range(len(Ls_forecast)):
            L = Ls_forecast[i]
            L_name = f"L{L}"
            for svd_method in svd_methods_forecast[i]:
                print(f"Data: {name}, Kind: {kind}, SVD-Method: {svd_method}, L: {L}...", end="")
                vforecast = compute_forecasts(series, [L], groups=forecast_groups, length=length,
                                              kind=kind, forecast_method="vector",
                                              svd_method=svd_method, neig=neig,
                                              column_projector=column_projector, row_projector=row_projector,
                                              **kwargs)
                pd.testing.assert_frame_equal(vforecast[L_name], test_data["vforecast"][L_name],
                                              rtol=tolerance)
                print("OK")


all_test_data = load_dataframes()
