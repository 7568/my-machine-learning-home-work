# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/14
Description:
"""


class IrisDataIter:
    """
    迭代器
    """

    def __init__(self, m, bach):

        self.data = m
        self.bach = bach
        self.length = len(m)
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.length - 1:
            self.index += 1
            _start = self.index * self.bach
            if _start > self.length:
                raise StopIteration
            _end = (self.index + 1) * self.bach
            if _end > self.length:
                _end = self.length
                # _start = _end - self.bach  # 最后一波直接取最后bach个元素
            _train_index = self.data[_start: _end]
            return _train_index
        else:
            raise StopIteration


