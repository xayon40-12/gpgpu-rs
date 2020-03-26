// each steps length for each respective dimensions must be constant.
pub struct DataFile {
    dx: Vec<f64>, // the difference between each steps for each dimensions.
    lenx: Vec<usize>,
    data: Vec<f64>
}

impl DataFile {
    fn from_column<'a>(text: &'a str) -> DataFile {
        let mut tab = text.trim().lines().map(|l| 
            l.trim().split(" ").map(|n| n.parse::<f64>().unwrap()).collect::<Vec<_>>()
        ).collect::<Vec<_>>();
        tab.sort_by(|a,b| {
            let mut i = 0;
            for i in 0..a.len()-1 {
                if a[i] < b[i] {
                    return std::cmp::Ordering::Less;
                } else if a[i] > b[i] {
                    return std::cmp::Ordering::Greater;
                }
            }
            std::cmp::Ordering::Equal
        });


        DataFile{ dx, lenx, data }
    }

    fn get(&self, coord: &[f64]) -> f64 {
        assert_eq!(coord.len(),self.dx.len());
        let mut idx = vec![];
        for i in 0..coord.len() {
            idx.push((coord[i] / self.dx[i]) as usize);
        }
        let mut id = idx[0];
        for i in 0..idx.len() {
            id *= self.lenx[i];
            id += idx[i];
        }
        self.data[id]
    }
}
