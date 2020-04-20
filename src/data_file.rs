use crate::functions::{Function,SFunction};
use crate::descriptors::{ConstructorTypes::CF64,FunctionConstructor::*};

// each steps length for each respective dimensions must be constant.
#[derive(Debug)]
pub struct DataFile {
    pub dx: Vec<f64>, // the difference between each steps for each dimensions.
    pub lenx: Vec<usize>,
    pub start: Vec<f64>,
    pub data: Vec<f64>,
}

pub enum Format<'a> {
    Column(&'a str),
}

impl DataFile {
    pub fn parse(from: Format) -> DataFile {
        match from {
            Format::Column(f) => from_column(f),
        }
    }

    pub fn get(&self, coords: &[f64]) -> f64 {
        let len = coords.len();
        assert_eq!(len,self.dx.len());
        let idx: Vec<_> = (0..len).map(|i| ((coords[i]-self.start[i]) / self.dx[i]) as usize).collect();
        let id = (0..len).fold(0, |a,i| a + idx[i]*self.lenx[i]);
        self.data[id]
    }

    pub fn to_function(&self, name: &str, huge: bool) -> SFunction {
        self.gen_func(name, format!("
            ulong idx[{len}];
            for(int i = 0; i<{len}; i++) {{
                idx[i] = (coords[i]-start[i]) / dx[i];
            }}
            ulong id = 0;
            for(int i = 0; i<{len}; i++) {{
                id += idx[i]*lenx[i];
            }}
            return data[id];
        ",
        len = self.dx.len()), huge)
    }

    pub fn get_interpolated(&self, coords: &[f64]) -> f64 {
        let len = coords.len();
        assert_eq!(len,self.dx.len());
        let xs: Vec<_> = (0..len).map(|i| {
            let a = (coords[i]-self.start[i]) / self.dx[i];
            let b = a as usize;
            (a-b as f64,b)
        }).collect();
        let mut ys: Vec<_> = (0..1<<len).map(|j|
            self.data[(0..len).fold(0, |a,i| a + (xs[i].1 + ((j>>i)&1))*self.lenx[i])]
        ).collect();
        for i in 0..len {
            for j in (0..1<<(len-i-1)).map(|r| r<<(i+1)) {
                ys[j] = ys[j] + (ys[j+(1<<i)]-ys[j])*xs[i].0;
            }
        }
        ys[0]
    }

    pub fn to_function_interpolated(&self, name: &str, huge: bool) -> SFunction {
        self.gen_func(name, format!("
            ulong xs[{len}];
            double xsd[{len}];
            for(int i = 0; i<{len}; i++) {{
                xsd[i] = (coords[i]-start[i]) / dx[i];
                xs[i] = xsd[i];
                xsd[i] -= xs[i];
            }}
            double ys[1<<{len}];
            for(int j = 0; j<(1<<{len}); j++) {{
                ulong id = 0;
                for(int i = 0; i<{len}; i++) {{
                    id += (xs[i]+((j>>i)&1))*lenx[i];
                }}
                ys[j] = data[id];
            }}
            for(int i = 0; i<{len}; i++) {{
                for(int j = 0; j<(1<<{len}); j+=(2<<i)) {{
                    ys[j] = ys[j] + (ys[j+(1<<i)]-ys[j])*xsd[i];
                }}
            }}
            return ys[0];
        ",
        len = self.dx.len()), huge)
    }

    pub fn gen_func(&self, name: &str, content: String, huge: bool) -> SFunction {
        if huge {
            let src = format!("
                double dx[] = {{{}}};
                ulong lenx[] = {{{}}};
                double start[] = {{{}}};

                {}
            ",
            self.dx.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
            self.lenx.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
            self.start.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
            content);

            (&Function {
                name: name,
                args: vec![FCPtr("coords",CF64),FCGlobalPtr("data",CF64)],
                ret_type: Some(CF64),
                src: &src,
                needed: vec![],
            }).into()
        } else {
            let src = format!("
                double dx[] = {{{}}};
                ulong lenx[] = {{{}}};
                double start[] = {{{}}};
                double data[] = {{{}}};

                {}
            ",
            self.dx.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
            self.lenx.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
            self.start.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
            self.data.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
            content);

            (&Function {
                name: name,
                args: vec![FCPtr("coords",CF64)],
                ret_type: Some(CF64),
                src: &src,
                needed: vec![],
            }).into()
        }
    }
}

fn from_column(text: &str) -> DataFile {
    let mut tab = text.trim().lines()
        .map(|l| l.trim())
        .filter(|l| l.len()>0 && !l.starts_with('#') && !l.starts_with("//"))
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

    DataFile{ dx, lenx, start, data}
}
