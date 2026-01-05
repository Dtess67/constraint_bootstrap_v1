import pytest
from constraint_bootstrap.partners.mixed_v1 import MixedPartnerV1

def test_mixed_partner():
    p = MixedPartnerV1()
    
    # sent=[2,2,3] -> len=3 prime => 5; sum=7 prime => 7 -> expect (5,7)
    assert p.respond((2, 2, 3)) == (5, 7)
    
    # sent=[2,2,4] -> len=3 prime => 5; sum=8 not prime -> expect (5,)
    assert p.respond((2, 2, 4)) == (5,)
    
    # sent=[1,1,1,1] -> len=4 not prime; sum=4 not prime -> expect ()
    assert p.respond((1, 1, 1, 1)) == ()

def test_mixed_empty():
    p = MixedPartnerV1()
    assert p.respond(()) == ()
