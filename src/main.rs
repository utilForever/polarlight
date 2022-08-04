use polarlight::Tensor;

#[rustfmt::skip]
fn main() {
    println!("Hello, world!");
    let t = Tensor::build(vec![2, 2, 3],
                          vec![
                              1, 2, 3,
                              4, 5, 6,
                              7, 8, 9,
                              10, 11, 12,
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


    let tmp3 = tmp1.add(&tmp2);
    let tmp4 = tmp1.sub(&tmp2);
    let tmp5 = tmp1.div(&tmp2);
    let tmp6 = tmp1.mul(&tmp2);
    let tmp7 = tmp1.matmul(&tmpmul);
    let tmp8 = tmp1.transpose();
    let tmp9 = tmp1.matmul(&tmp2.transpose());
    let tmp_reshape = t.reshape(vec![12]);
    println!("{:?}", tmp3.get(vec![1, 1]));
    println!("{:?}", tmp4.get(vec![1, 1]));
    println!("{:?}", tmp5.get(vec![1, 1]));
    t.print();
    tmp3.print();
    tmp4.print();
    tmp5.print();
    tmp6.print();
    tmp7.print();
    (t.add(&t)).print();
    tmp8.print();
    tmp9.print();
    tmp_reshape.print();
}
