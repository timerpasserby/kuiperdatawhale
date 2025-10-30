//
// Created by fss on 23-5-28.
//
#include <armadillo>
#include <glog/logging.h>
#include <gtest/gtest.h>
void Axby(const arma::fmat &x, const arma::fmat &w, const arma::fmat &b,
          arma::fmat &y)
{
  // 把代码写这里 完成y = w * x + b的运算
  // if (x.empty() || w.empty())
  // {
  //   return;
  // }
  // const arma::uword w_row = w.n_rows;
  // const arma::uword w_col = w.n_cols;
  // const arma::uword x_row = x.n_rows;
  // const arma::uword x_col = x.n_cols;
  // const arma::uword b_row = b.n_rows;
  // const arma::uword b_col = b.n_cols;
  // // m * n  n * k
  // if (w_col != x_row)
  // {
  //   printf("w & x does not match !!!!");
  //   return;
  // }
  // if (w_row != b_row && x_col != b_col)
  // {
  //   printf("b's dims dose not match!!");
  // }
  // // 两个for循环解决矩阵乘法
  // // 为什么用size_t？而不用int？
  // for (size_t i = 0; i < w_row; ++i)
  // {
  //   float tmp = 0.0f;
  //   for (size_t j = 0; j < x_col; j++)
  //   {

  //     for (size_t k = 0; k < w_col; k++)
  //     {
  //       tmp += w(i, k) * x(k, j);
  //     }
  //     y(i, j) = tmp + b(i, j);
  //   }
  // }
  y = w * x + b;
}

TEST(test_arma, Axby)
{
  using namespace arma;
  fmat w = "1,2,3;"
           "4,5,6;"
           "7,8,9;";

  fmat x = "1,2,3;"
           "4,5,6;"
           "7,8,9;";

  fmat b = "1,1,1;"
           "2,2,2;"
           "3,3,3;";

  fmat answer = "31,37,43;"
                "68,83,98;"
                "105,129,153";

  fmat y;
  Axby(x, w, b, y);
  ASSERT_EQ(approx_equal(y, answer, "absdiff", 1e-5f), true);
}

void EPowerMinus(const arma::fmat &x, arma::fmat &y)
{
  // 把代码写这里 完成y = e^{-x}的运算
  y = arma::exp(-x);
}

TEST(test_arma, EPowerMinus)
{
  using namespace arma;

  fmat x(224, 224);
  x.randu();

  fmat y;
  EPowerMinus(x, y);
  std::vector<float> x1(x.mem, x.mem + 224 * 224);
  ASSERT_EQ(y.empty(), false);
  for (int i = 0; i < 224 * 224; ++i)
  {
    ASSERT_LE(std::abs(std::exp(-x1.at(i)) - y.at(i)), 1e-5f);
  }
}

void Axpy(const arma::fmat &x, arma::fmat &Y, float a, float y)
{
  // 编写Y = a * x + y
  Y = a * x + y;
}

TEST(test_arma, axpy)
{
  using namespace arma;
  fmat x(224, 224);
  x.randu();

  fmat Y;
  float a = 3.f;
  float y = 4.f;
  Axpy(x, Y, a, y);

  ASSERT_EQ(Y.empty(), false);
  std::vector<float> x1(x.mem, x.mem + 224 * 224);
  for (int i = 0; i < 224 * 224; ++i)
  {
    ASSERT_LE(std::abs(x.at(i) * a + y - Y.at(i)), 1e-5f);
  }
}