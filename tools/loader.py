# -*- coding: UTF-8 -*-
import numpy as np
import xlrd

""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-June-15.
Load datasets from .xlsx file.
"""
################################################
# null  # Variables  # Objectives
# 1     # vars       # objs
################################################
# for exp2_plus.
def load_XY(path, n_vars, n_objs, n_samples):
    src_file = xlrd.open_workbook(path)
    src_sheets = src_file.sheets()
    src_sheet = src_sheets[0]
    X = np.zeros((n_samples, n_vars), dtype=float)
    Y = np.zeros((n_samples, n_objs), dtype=float)
    for index in range(n_samples):
        row_data = src_sheet.row_values(index + 1)
        vars_end = n_vars + 1
        X[index] = row_data[1:vars_end]
        Y[index] = row_data[vars_end:vars_end + n_objs]
    Y = np.around(Y, decimals=4)
    return X, Y

################################################
# null
# 1     # objs      # null      # vars
################################################
# exp2 and exp3
def load_XY_for_exp2(path, n_vars, n_objs, n_samples):
    src_file = xlrd.open_workbook(path)
    src_sheets = src_file.sheets()
    src_sheet = src_sheets[0]
    X = np.zeros((n_samples, n_vars), dtype=float)
    Y = np.zeros((n_samples, n_objs), dtype=float)
    for index in range(n_samples):
        row_data = src_sheet.row_values(index + 1)
        X[index] = row_data[n_objs + 2:]
        Y[index] = row_data[1:n_objs + 1]
    Y = np.around(Y, decimals=4)
    return X, Y