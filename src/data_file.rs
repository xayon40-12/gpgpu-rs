// each steps length for each respective dimensions must be constant.
#[derive(Debug)]
pub struct DataFile {
    dx: Vec<f64>, // the difference between each steps for each dimensions.
    lenx: Vec<usize>,
    start: Vec<f64>,
    data: Vec<f64>
}

impl DataFile {
    pub fn from_column<'a>(text: &'a str) -> DataFile {
        let mut tab = text.trim().lines()
            .map(|l| l.trim())
            .filter(|l| l.len()>0)
            .map(|l| l.split(" ").map(|n| n.parse::<f64>().expect(&format!("Could not parse f64 in invocation of DataFile::from_column."))).collect::<Vec<_>>()
        ).collect::<Vec<_>>();
        if tab.len() < 2 {
            panic!("Data file should contains at least 2 lines.");
        }
        tab.sort_by(|a,b| {
            for i in 0..a.len()-1 {
                if a[i] < b[i] {
                    return std::cmp::Ordering::Less;
                } else if a[i] > b[i] {
                    return std::cmp::Ordering::Greater;
                }
            }
            std::cmp::Ordering::Equal
        });
        let len = tab[0].len()-1;
        let mut r = len-1;

        let mut dx: Vec<f64> = vec![0.0;len];
        let mut lenx: Vec<usize> = vec![tab.len();len];
        let data: Vec<f64> = tab.iter().map(|l| l[len]).collect();

        let mut tab = tab.into_iter();
        let start = tab.next().unwrap();

        for (i,l) in tab.enumerate() {
            let i = i+1;
            if l.len() != len+1 {
                panic!("Each lines should contains the same amount of number in DataFile::from_column.");
            }
            if r != len && l[r] != start[r] {
                dx[r] = l[r]-start[r];
                lenx[r] = i;
                if r > 0 {
                    r -= 1;
                } else {
                    r = len;
                }
            }
            for c in r..len {
                let lx= if c > 0 { lenx[c-1] } else { std::usize::MAX };
                if l[c] != start[c] + ((i/lenx[c])%lx) as f64*dx[c] {
                    panic!("The difference between 2 consecutive coordinate should be constant for each dimensions.");
                }
            }
        }

        DataFile{ dx, lenx, start, data }
    }

    pub fn get(&self, coord: &[f64]) -> f64 {
        let len = coord.len();
        assert_eq!(len,self.dx.len());
        let mut idx = vec![];
        for i in 0..coord.len() {
            idx.push(((coord[i]-self.start[i]) / self.dx[i]) as usize);
        }
        let mut id = 0;
        for i in 0..idx.len() {
            id += idx[i]*self.lenx[i];
        }
        self.data[id]
    }
}
