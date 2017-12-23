// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::stable_hasher;
use serialize::{self, Decoder, Encoder};

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy, RustcEncodable, RustcDecodable)]
pub struct Fingerprint(u64, u64);

impl Fingerprint {

    pub const ZERO: Fingerprint = Fingerprint(0, 0);

    #[inline]
    pub fn from_smaller_hash(hash: u64) -> Fingerprint {
        Fingerprint(hash, hash)
    }

    #[inline]
    pub fn to_smaller_hash(&self) -> u64 {
        self.0
    }

    #[inline]
    pub fn as_value(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    #[inline]
    pub fn combine(self, other: Fingerprint) -> Fingerprint {
        // See https://stackoverflow.com/a/27952689 on why this function is
        // implemented this way.
        Fingerprint(
            self.0.wrapping_mul(3).wrapping_add(other.0),
            self.1.wrapping_mul(3).wrapping_add(other.1)
        )
    }

    pub fn to_hex(&self) -> String {
        format!("{:x}{:x}", self.0, self.1)
    }

}

impl ::std::fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(formatter, "{:x}-{:x}", self.0, self.1)
    }
}

impl stable_hasher::StableHasherResult for Fingerprint {
    fn finish(hasher: stable_hasher::StableHasher<Self>) -> Self {
        let (_0, _1) = hasher.finalize();
        Fingerprint(_0, _1)
    }
}

impl<CTX> stable_hasher::HashStable<CTX> for Fingerprint {
    #[inline]
    fn hash_stable<W: stable_hasher::StableHasherResult>(&self,
                                          _: &mut CTX,
                                          hasher: &mut stable_hasher::StableHasher<W>) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl serialize::UseSpecializedEncodable for Fingerprint {
    fn default_encode<E: Encoder>(&self, _e: &mut E) -> Result<(), E::Error> {
        panic!("hit");
    }
}

impl serialize::UseSpecializedDecodable for Fingerprint {
    fn default_decode<D: Decoder>(_d: &mut D) -> Result<Fingerprint, D::Error> {
        panic!("hit");
    }
}

impl<'a> serialize::SpecializedEncoder<Fingerprint> for serialize::opaque::Encoder<'a> {
    fn specialized_encode(&mut self, _f: &Fingerprint) -> Result<(), Self::Error> {
        panic!("hit");
    }
}

impl<'a> serialize::SpecializedDecoder<Fingerprint> for serialize::opaque::Decoder<'a> {
    fn specialized_decode(&mut self) -> Result<Fingerprint, Self::Error> {
        panic!("hit");
    }
}
