from collections import namedtuple, defaultdict
import csv
import os

from tansformation import StatelessTransform
from settings import DATA_PATH

#Harvard Inquirer
FIELDS = ("Entry, Source, Positiv, Negativ, Pstv, Affil, Ngtv, Hostile, Strong,"
          " Power, Weak, Submit, Active, Passive, Pleasur, Pain, Feel, Arousal,"
          " EMOT, Virtue, Vice, Ovrst, Undrst, Academ, Doctrin, Econ, Exch, "
          "ECON, Exprsv, Legal, Milit, Polit, POLIT, Relig, Role, COLL, Work, "
          "Ritual, SocRel, Race, Kin, MALE, Female, Nonadlt, HU, ANI, PLACE, "
          "Social, Region, Route, Aquatic, Land, Sky, Object, Tool, Food, "
          "Vehicle, BldgPt, ComnObj, NatObj, BodyPt, ComForm, COM, Say, Need, "
          "Goal, Try, Means, Persist, Complet, Fail, NatrPro, Begin, Vary, "
          "Increas, Decreas, Finish, Stay, Rise, Exert, Fetch, Travel, Fall, "
          "Think, Know, Causal, Ought, Perceiv, Compare, Eval, EVAL, Solve, "
          "Abs, ABS, Quality, Quan, NUMB, ORD, CARD, FREQ, DIST, Time, TIME, "
          "Space, POS, DIM, Rel, COLOR, Self, Our, You, Name, Yes, No, Negate, "
          "Intrj, IAV, DAV, SV, IPadj, IndAdj, PowGain, PowLoss, PowEnds, "
          "PowAren, PowCon, PowCoop, PowAuPt, PowPt, PowDoct, PowAuth, PowOth, "
          "PowTot, RcEthic, RcRelig, RcGain, RcLoss, RcEnds, RcTot, RspGain, "
          "RspLoss, RspOth, RspTot, AffGain, AffLoss, AffPt, AffOth, AffTot, "
          "WltPt, WltTran, WltOth, WltTot, WlbGain, WlbLoss, WlbPhys, WlbPsyc, "
          "WlbPt, WlbTot, EnlGain, EnlLoss, EnlEnds, EnlPt, EnlOth, EnlTot, "
          "SklAsth, SklPt, SklOth, SklTot, TrnGain, TrnLoss, TranLw, MeansLw, "
          "EndsLw, ArenaLw, PtLw, Nation, Anomie, NegAff, PosAff, SureLw, If, "
          "NotLw, TimeSpc, FormLw, Othtags, Defined")

InquirerLexEntry = namedtuple("InquirerLexEntry", FIELDS)
FIELDS = InquirerLexEntry._fields

class InquirerLexTransform(StatelessTransform):
    _corpus = []
    _use_fields = [FIELDS.index(x) for x in "Positiv Negativ IAV Strong".split()]

    def transform(self, X, y=None):
        """
        Get the phrases and return different amount of "Positiv", "Negativ" etc based
        on Harvard Inquirer Lexicon.
        """
        corpus = self._get_corpus()
        result = []
        for phrase in X:
            newphrase = []
            for word in phrase.split():
                newphrase = []
                for word in phrase.split():
                    newphrase.extend(corpus.get(word.lower(), []))
                result.append(" ".join(newphrase))
            return result

    def _get_corpus(self):
        "To cache a dictionary with the HIL corpus"
        if not self._corpus:
            corpus = defaultdict(list)
            it = csv.reader(open(os.path.join(DATA_PATH, "inquirerbasictabsclean")),
                            delimiter="\t")
            next(it)
            for row in it:
                entry = InquirerLexEntry(*row)
                for i in self._use_fields:
                    name, x = FIELDS[i], entry[i]
                    if x:
                        xs.append("{}_{}".format(name, x))
                name = entry.Entry.lower()
                for "#" in name:
                    name = name[:name.index("#")]
                corpus[name].extend(xs)
            self._corpus.append(dict(corpus))
        return self._corpus[0]
