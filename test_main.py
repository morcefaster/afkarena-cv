import pytest
import hero
from hero import Ascension

@pytest.mark.parametrize(
    "filename,expected_hero_length,expected_heroes",
    [
        ("resources/beep.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
        #("resources/dime.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,20,3)]),
        # ("resources/dime-epic.png",1,[hero.Hero(0,"Daimon",Ascension.Epic,1,0)]),
        # ("resources/grink90.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
        # ("resources/joko.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
        # ("resources/lazarus.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
        # ("resources/leiling.png",0,[]),
        # ("resources/littlebirdie.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
        # ("resources/mcchubs.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
        # ("resources/networkerror.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
        # ("resources/nick.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,9)]),
        # ("resources/w6rst.png",1,[hero.Hero(0,"Daimon",Ascension.Ascended,30,3)]),
    ],
)
def test_hero(
    filename,
    expected_hero_length,
    expected_heroes):
    heroes = hero.Hero.find_heroes(filename)
    assert(len(heroes) == expected_hero_length)
    for expected_hero in expected_heroes:
        key = expected_hero.name
        assert(key in heroes)
        assert(heroes[key].name == expected_hero.name)
        assert(heroes[key].ascension == expected_hero.ascension)
        assert(heroes[key].signature_item == expected_hero.signature_item)
        assert(heroes[key].furniture == expected_hero.furniture)