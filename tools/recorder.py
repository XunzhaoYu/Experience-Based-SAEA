# -*- coding: UTF-8 -*-
import numpy as np
import xlwt


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
Record optimization results in .xlsx file.
Designed for multi-objective optimization problems. No constraint.
"""
class Recorder:
    def __init__(self, sheet_name='sheet1'):
        self.record_file = xlwt.Workbook()
        self.record_sheet = self.record_file.add_sheet(sheetname=sheet_name)
        # declare n_vars and n_objs
        self.n_vars = None
        self.n_objs = None
        self.style = xlwt.XFStyle()
        self.style.num_format_str = '0.0000'

    # record initial archive (X), fitness (Y), and performance.
    def init(self, X, Y, performance_list, performance_name_list):
        size_archive, self.n_vars = np.shape(X)
        self.n_objs = np.size(Y, 1)

        # initialize titles.
        self.record_sheet.write(0, 1, 'Variables')
        self.record_sheet.write(0, self.n_vars + 1, 'Objectives')
        for performance_index in range(len(performance_list)):
            self.record_sheet.write(0, self.n_vars + self.n_objs + performance_index + 1, performance_name_list[performance_index])

        # initialize data
        for ind_index in range(size_archive-1):
            self._write_data(ind_index + 1, X[ind_index], Y[ind_index])
        self.write(size_archive, X[-1], Y[-1], performance_list)

    # record one evaluated solution (x, y) and its performance.
    def write(self, row_index, x, y, performance_list):
        self._write_data(row_index, x, y)
        # write performance
        for index in range(len(performance_list)):
            self.record_sheet.write(row_index, self.n_vars + self.n_objs + index + 1, performance_list[index], self.style)

    # record one evaluated solution (x, y):
    def _write_data(self, row_index, x, y):
        # write the row index
        self.record_sheet.write(row_index, 0, row_index)
        # write variables
        for dim_index in range(self.n_vars):
            self.record_sheet.write(row_index, dim_index + 1, x[dim_index], self.style)
        # write objectives
        for dim_index in range(self.n_objs):
            self.record_sheet.write(row_index, self.n_vars + dim_index + 1, y[dim_index], self.style)

    # save record
    def save(self, name):
        self.record_file.save(name)


