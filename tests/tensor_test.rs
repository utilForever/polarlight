use polarlight::Tensor;

#[rustfmt::skip]
#[test]
fn test_tensor() {
    let t = Tensor::build(vec![2, 2, 3],
                          vec![
                              1.0, 2.0, 3.0,
                              4.0, 5.0, 6.0,
                              7.0, 8.0, 9.0,
                              10.0, 11.0, 12.0,
                          ]);
    let tmp1 = Tensor::build(vec![2, 3],
                             vec![
                                 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                             ]);
    let tmp2 = Tensor::build(vec![2, 3],
                             vec![
                                 7.0, 8.0, 9.0,
                                 10.0, 11.0, 12.0,
                             ]);
    let tmpmul = Tensor::build(vec![3, 4],
                               vec![
                                   1.0, 2.0, 3.0, 4.0,
                                   5.0, 6.0, 7.0, 8.0,
                                   9.0, 10.0, 11.0, 12.0,
                               ]);

    let tmp_add = tmp1.add(&tmp2);
    tmp_add.print();

    let tmp_sub = tmp1.sub(&tmp2);
    tmp_sub.print();

    let tmp_div = tmp1.div(&tmp2);
    tmp_div.print();

    let tmp_mul = tmp1.mul(&tmp2);
    tmp_mul.print();

    let tmp_matmul = tmp1.matmul(&tmpmul);
    tmp_matmul.print();

    let tmp_transpose = tmp1.transpose();
    tmp_transpose.print();

    let tmp_matmul_transpose = tmp1.matmul(&tmp2.transpose());
    tmp_matmul_transpose.print();

    let tmp_reshape = t.reshape(vec![12]);
    tmp_reshape.print();
}
