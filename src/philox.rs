pub fn philox2x64(mut counter: [u64;2], mut key: u64, rounds: u8) -> [u64;2] {
    for _ in 0..rounds {
        let prod = (0xD2B74407B1CE6E93 as u128).wrapping_mul(counter[0] as u128);
        let hi = (prod >> 64) as u64;
        let lo = prod as u64;
        counter[0] = hi^key^counter[1];
        counter[1] = lo;
        key = key.wrapping_add(0x9E3779B97F4A7C15);
    }
    counter
}

pub fn philox4x64(mut counter: [u64;4], mut key: [u64;2], rounds: u8) -> [u64;4] {
    for _ in 0..rounds {
        let prod = (0xD2B74407B1CE6E93 as u128).wrapping_mul(counter[0] as u128);
        let hi0 = (prod >> 64) as u64;
        let lo0 = prod as u64;
        let prod = (0xCA5A826395121157 as u128).wrapping_mul(counter[2] as u128);
        let hi1 = (prod >> 64) as u64;
        let lo1 = prod as u64;
        counter[0] = hi1^key[1]^counter[3];
        counter[1] = lo0;
        counter[2] = hi0^key[0]^counter[1];
        counter[3] = lo1;
        key[0] = key[0].wrapping_add(0x9E3779B97F4A7C15);
        key[1] = key[1].wrapping_add(0xBB67AE8584CAA73B);
    }
    counter
}

pub fn philox4x32(mut counter: [u32;4], mut key: [u32;2], rounds: u8) -> [u32;4] {
    for _ in 0..rounds {
        let prod = (0xD2511F53 as u64).wrapping_mul(counter[0] as u64);
        let hi0 = (prod >> 32) as u32;
        let lo0 = prod as u32;
        let prod = (0xCD9E8D57 as u64).wrapping_mul(counter[2] as u64);
        let hi1 = (prod >> 32) as u32;
        let lo1 = prod as u32;
        counter[0] = hi1^key[1]^counter[3];
        counter[1] = lo0;
        counter[2] = hi0^key[0]^counter[1];
        counter[3] = lo1;
        key[0] = key[0].wrapping_add(0x9E3779B9);
        key[1] = key[1].wrapping_add(0xBB67AE85);
    }
    counter
}
