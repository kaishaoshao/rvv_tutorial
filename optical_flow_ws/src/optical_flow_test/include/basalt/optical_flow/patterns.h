/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <Eigen/Dense>

namespace basalt {

template <class Scalar>
struct Pattern24 {
  //          00  01
  //
  //      02  03  04  05
  //
  //  06  07  08  09  10  11
  //
  //  12  13  14  15  16  17
  //
  //      18  19  20  21
  //
  //          22  23
  //
  // -----> x
  // |
  // |
  // y

  static constexpr Scalar pattern_raw[][2] = {
      {-1, 5},  {1, 5},

      {-3, 3},  {-1, 3},  {1, 3},   {3, 3},

      {-5, 1},  {-3, 1},  {-1, 1},  {1, 1},  {3, 1},  {5, 1},

      {-5, -1}, {-3, -1}, {-1, -1}, {1, -1}, {3, -1}, {5, -1},

      {-3, -3}, {-1, -3}, {1, -3},  {3, -3},

      {-1, -5}, {1, -5}

  };

  static constexpr int PATTERN_SIZE =
      sizeof(pattern_raw) / (2 * sizeof(Scalar));

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern24<Scalar>::Matrix2P Pattern24<Scalar>::pattern2 =
    Eigen::Map<Pattern24<Scalar>::Matrix2P>((Scalar *)
                                                Pattern24<Scalar>::pattern_raw);

template <class Scalar>
struct Pattern52 {
  //          00  01  02  03
  //
  //      04  05  06  07  08  09
  //
  //  10  11  12  13  14  15  16  17
  //
  //  18  19  20  21  22  23  24  25
  //
  //  26  27  28  29  30  31  32  33
  //
  //  34  35  36  37  38  39  40  41
  //
  //      42  43  44  45  46  47
  //
  //          48  49  50  51
  //
  // -----> x
  // |
  // |
  // y

  static constexpr Scalar pattern_raw[][2] = {
      {-3, 7},  {-1, 7},  {1, 7},   {3, 7},

      {-5, 5},  {-3, 5},  {-1, 5},  {1, 5},   {3, 5},  {5, 5},

      {-7, 3},  {-5, 3},  {-3, 3},  {-1, 3},  {1, 3},  {3, 3},
      {5, 3},   {7, 3},

      {-7, 1},  {-5, 1},  {-3, 1},  {-1, 1},  {1, 1},  {3, 1},
      {5, 1},   {7, 1},

      {-7, -1}, {-5, -1}, {-3, -1}, {-1, -1}, {1, -1}, {3, -1},
      {5, -1},  {7, -1},

      {-7, -3}, {-5, -3}, {-3, -3}, {-1, -3}, {1, -3}, {3, -3},
      {5, -3},  {7, -3},

      {-5, -5}, {-3, -5}, {-1, -5}, {1, -5},  {3, -5}, {5, -5},

      {-3, -7}, {-1, -7}, {1, -7},  {3, -7}

  };

  static constexpr int PATTERN_SIZE =
      sizeof(pattern_raw) / (2 * sizeof(Scalar));

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2; // pattern2表示的是2*52的矩阵
};

template <class Scalar>
const typename Pattern52<Scalar>::Matrix2P Pattern52<Scalar>::pattern2 =
    Eigen::Map<Pattern52<Scalar>::Matrix2P>((Scalar *)
                                                Pattern52<Scalar>::pattern_raw);

// Same as Pattern52 but twice smaller
template <class Scalar>
struct Pattern51 { // Pattern51跟Pattern52一样，只是patch只有一半那么小。换句话说patch点个数一样多，只是点更紧密
  static constexpr int PATTERN_SIZE = Pattern52<Scalar>::PATTERN_SIZE;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2; // pattern2表示的是2*52的矩阵
};

template <class Scalar>
const typename Pattern51<Scalar>::Matrix2P Pattern51<Scalar>::pattern2 =
    0.5 * Pattern52<Scalar>::pattern2; // 偏移量只有0.5倍

// Same as Pattern52 but 0.75 smaller
template <class Scalar>
struct Pattern50 {
  static constexpr int PATTERN_SIZE = Pattern52<Scalar>::PATTERN_SIZE;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern50<Scalar>::Matrix2P Pattern50<Scalar>::pattern2 =
    0.75 * Pattern52<Scalar>::pattern2;

// wxliu on 2025-1-7
template <class Scalar>
struct Pattern441 {
  //  000  001  002  003  004  005  006  007  008  009  010  011  012  013  014  015  016  017  018  019  020
  //  021  022  023  024  025  026  027  028  029  030  031  032  033  034  035  036  037  038  039  040  041
  //  042  043  044  045  046  047  048  049  050  051  052  053  054  055  056  057  058  059  060  061  062
  //  063  064  065  066  067  068  069  070  071  072  073  074  075  076  077  078  079  080  081  082  083
  //  084  085  086  087  088  089  090  091  092  093  094  095  096  097  098  099  100  101  102  103  104
  //  105  106  107  108  109  110  111  112  113  114  115  116  117  118  119  120  121  122  123  124  125
  //  126  127  128  129  130  131  132  133  134  135  136  137  138  139  140  141  142  143  144  145  146
  //  147  148  149  150  151  152  153  154  155  156  157  158  159  160  161  162  163  164  165  166  167
  //  168  169  170  171  172  173  174  175  176  177  178  179  180  181  182  183  184  185  186  187  188
  //  189  190  191  192  193  194  195  196  197  198  199  200  201  202  203  204  205  206  207  208  209
  //  210  211  212  213  214  215  216  217  218  219 [220] 221  222  223  224  225  226  227  228  229  230
  //  231  232  233  234  235  236  237  238  239  240  241  242  243  244  245  246  247  248  249  250  251
  //  252  253  254  255  256  257  258  259  260  261  262  263  264  265  266  267  268  269  270  271  272
  //  273  274  275  276  277  278  279  280  281  282  283  284  285  286  287  288  289  290  291  292  293
  //  294  295  296  297  298  299  300  301  302  303  304  305  306  307  308  309  310  311  312  313  314
  //  315  316  317  318  319  320  321  322  323  324  325  326  327  328  329  330  331  332  333  334  335
  //  336  337  338  339  340  341  342  343  344  345  346  347  348  349  350  351  352  353  354  355  356
  //  357  358  359  360  361  362  363  364  365  366  367  368  369  370  371  372  373  374  375  376  377
  //  378  379  380  381  382  383  384  385  386  387  388  389  390  391  392  393  394  395  396  397  398
  //  399  400  401  402  403  404  405  406  407  408  409  410  411  412  413  414  415  416  417  418  419
  //  420  421  422  423  424  425  426  427  428  429  430  431  432  433  434  435  436  437  438  439  440
  //
  // '[center]' is the postion of a point which will be tracked.
  //
  // -----> x
  // |
  // |
  // y

  static constexpr Scalar pattern_raw[][2] = {
      {-10, -10}, {-9, -10}, {-8, -10}, {-7, -10}, {-6, -10}, {-5, -10}, {-4, -10}, {-3, -10}, {-2, -10}, {-1, -10}, {0, -10}, {1, -10}, {2, -10}, {3, -10}, {4, -10}, {5, -10}, {6, -10}, {7, -10}, {8, -10}, {9, -10}, {10, -10}, 
      {-10, -9}, {-9, -9}, {-8, -9}, {-7, -9}, {-6, -9}, {-5, -9}, {-4, -9}, {-3, -9}, {-2, -9}, {-1, -9}, {0, -9}, {1, -9}, {2, -9}, {3, -9}, {4, -9}, {5, -9}, {6, -9}, {7, -9}, {8, -9}, {9, -9}, {10, -9}, 
      {-10, -8}, {-9, -8}, {-8, -8}, {-7, -8}, {-6, -8}, {-5, -8}, {-4, -8}, {-3, -8}, {-2, -8}, {-1, -8}, {0, -8}, {1, -8}, {2, -8}, {3, -8}, {4, -8}, {5, -8}, {6, -8}, {7, -8}, {8, -8}, {9, -8}, {10, -8}, 
      {-10, -7}, {-9, -7}, {-8, -7}, {-7, -7}, {-6, -7}, {-5, -7}, {-4, -7}, {-3, -7}, {-2, -7}, {-1, -7}, {0, -7}, {1, -7}, {2, -7}, {3, -7}, {4, -7}, {5, -7}, {6, -7}, {7, -7}, {8, -7}, {9, -7}, {10, -7}, 
      {-10, -6}, {-9, -6}, {-8, -6}, {-7, -6}, {-6, -6}, {-5, -6}, {-4, -6}, {-3, -6}, {-2, -6}, {-1, -6}, {0, -6}, {1, -6}, {2, -6}, {3, -6}, {4, -6}, {5, -6}, {6, -6}, {7, -6}, {8, -6}, {9, -6}, {10, -6}, 
      {-10, -5}, {-9, -5}, {-8, -5}, {-7, -5}, {-6, -5}, {-5, -5}, {-4, -5}, {-3, -5}, {-2, -5}, {-1, -5}, {0, -5}, {1, -5}, {2, -5}, {3, -5}, {4, -5}, {5, -5}, {6, -5}, {7, -5}, {8, -5}, {9, -5}, {10, -5}, 
      {-10, -4}, {-9, -4}, {-8, -4}, {-7, -4}, {-6, -4}, {-5, -4}, {-4, -4}, {-3, -4}, {-2, -4}, {-1, -4}, {0, -4}, {1, -4}, {2, -4}, {3, -4}, {4, -4}, {5, -4}, {6, -4}, {7, -4}, {8, -4}, {9, -4}, {10, -4}, 
      {-10, -3}, {-9, -3}, {-8, -3}, {-7, -3}, {-6, -3}, {-5, -3}, {-4, -3}, {-3, -3}, {-2, -3}, {-1, -3}, {0, -3}, {1, -3}, {2, -3}, {3, -3}, {4, -3}, {5, -3}, {6, -3}, {7, -3}, {8, -3}, {9, -3}, {10, -3}, 
      {-10, -2}, {-9, -2}, {-8, -2}, {-7, -2}, {-6, -2}, {-5, -2}, {-4, -2}, {-3, -2}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {3, -2}, {4, -2}, {5, -2}, {6, -2}, {7, -2}, {8, -2}, {9, -2}, {10, -2}, 
      {-10, -1}, {-9, -1}, {-8, -1}, {-7, -1}, {-6, -1}, {-5, -1}, {-4, -1}, {-3, -1}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, {3, -1}, {4, -1}, {5, -1}, {6, -1}, {7, -1}, {8, -1}, {9, -1}, {10, -1}, 
      {-10, 0}, {-9, 0}, {-8, 0}, {-7, 0}, {-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, 
      {-10, 1}, {-9, 1}, {-8, 1}, {-7, 1}, {-6, 1}, {-5, 1}, {-4, 1}, {-3, 1}, {-2, 1}, {-1, 1}, {0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1}, {8, 1}, {9, 1}, {10, 1}, 
      {-10, 2}, {-9, 2}, {-8, 2}, {-7, 2}, {-6, 2}, {-5, 2}, {-4, 2}, {-3, 2}, {-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2}, {5, 2}, {6, 2}, {7, 2}, {8, 2}, {9, 2}, {10, 2}, 
      {-10, 3}, {-9, 3}, {-8, 3}, {-7, 3}, {-6, 3}, {-5, 3}, {-4, 3}, {-3, 3}, {-2, 3}, {-1, 3}, {0, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 3}, {5, 3}, {6, 3}, {7, 3}, {8, 3}, {9, 3}, {10, 3}, 
      {-10, 4}, {-9, 4}, {-8, 4}, {-7, 4}, {-6, 4}, {-5, 4}, {-4, 4}, {-3, 4}, {-2, 4}, {-1, 4}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4, 4}, {5, 4}, {6, 4}, {7, 4}, {8, 4}, {9, 4}, {10, 4}, 
      {-10, 5}, {-9, 5}, {-8, 5}, {-7, 5}, {-6, 5}, {-5, 5}, {-4, 5}, {-3, 5}, {-2, 5}, {-1, 5}, {0, 5}, {1, 5}, {2, 5}, {3, 5}, {4, 5}, {5, 5}, {6, 5}, {7, 5}, {8, 5}, {9, 5}, {10, 5}, 
      {-10, 6}, {-9, 6}, {-8, 6}, {-7, 6}, {-6, 6}, {-5, 6}, {-4, 6}, {-3, 6}, {-2, 6}, {-1, 6}, {0, 6}, {1, 6}, {2, 6}, {3, 6}, {4, 6}, {5, 6}, {6, 6}, {7, 6}, {8, 6}, {9, 6}, {10, 6}, 
      {-10, 7}, {-9, 7}, {-8, 7}, {-7, 7}, {-6, 7}, {-5, 7}, {-4, 7}, {-3, 7}, {-2, 7}, {-1, 7}, {0, 7}, {1, 7}, {2, 7}, {3, 7}, {4, 7}, {5, 7}, {6, 7}, {7, 7}, {8, 7}, {9, 7}, {10, 7}, 
      {-10, 8}, {-9, 8}, {-8, 8}, {-7, 8}, {-6, 8}, {-5, 8}, {-4, 8}, {-3, 8}, {-2, 8}, {-1, 8}, {0, 8}, {1, 8}, {2, 8}, {3, 8}, {4, 8}, {5, 8}, {6, 8}, {7, 8}, {8, 8}, {9, 8}, {10, 8}, 
      {-10, 9}, {-9, 9}, {-8, 9}, {-7, 9}, {-6, 9}, {-5, 9}, {-4, 9}, {-3, 9}, {-2, 9}, {-1, 9}, {0, 9}, {1, 9}, {2, 9}, {3, 9}, {4, 9}, {5, 9}, {6, 9}, {7, 9}, {8, 9}, {9, 9}, {10, 9}, 
      {-10, 10}, {-9, 10}, {-8, 10}, {-7, 10}, {-6, 10}, {-5, 10}, {-4, 10}, {-3, 10}, {-2, 10}, {-1, 10}, {0, 10}, {1, 10}, {2, 10}, {3, 10}, {4, 10}, {5, 10}, {6, 10}, {7, 10}, {8, 10}, {9, 10}, {10, 10}

  };

  static constexpr int PATTERN_SIZE =
      sizeof(pattern_raw) / (2 * sizeof(Scalar));

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern441<Scalar>::Matrix2P Pattern441<Scalar>::pattern2 =
    Eigen::Map<Pattern441<Scalar>::Matrix2P>((Scalar *)
                                                Pattern441<Scalar>::pattern_raw);
// the end.

}  // namespace basalt
