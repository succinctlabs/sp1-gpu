#pragma once

#include "../../fields/bb31_t.cuh"

namespace poseidon2_bb31_16 {

namespace constants {

    constexpr int DIGEST_WIDTH = 8;
    constexpr int RATE = 8;
    constexpr int WIDTH = 16;
    constexpr int ROUNDS_P = 13;
    constexpr int ROUNDS_F = 8;
    constexpr int D = 7;

    __constant__ bb31_t INTERNAL_ROUND_CONSTANTS[ROUNDS_P] = {
        bb31_t(1196780786),
        bb31_t(36046858),
        bb31_t(1492041470),
        bb31_t(1864954859),
        bb31_t(883677154),
        bb31_t(1765843422),
        bb31_t(1418914503),
        bb31_t(1404382774),
        bb31_t(859661334),
        bb31_t(1548195514),
        bb31_t(104929687),
        bb31_t(178643863),
        bb31_t(1619872446)
    };

    __constant__ bb31_t EXTERNAL_ROUND_CONSTANTS[ROUNDS_F][WIDTH] = {
        {bb31_t(96748292),
         bb31_t(1951698684),
         bb31_t(177396853),
         bb31_t(719730562),
         bb31_t(640767983),
         bb31_t(1390633215),
         bb31_t(1716033721),
         bb31_t(1606702601),
         bb31_t(1746607367),
         bb31_t(1466015491),
         bb31_t(1498308946),
         bb31_t(831109173),
         bb31_t(1029197920),
         bb31_t(1969905919),
         bb31_t(83412884),
         bb31_t(1911782445)},
        {bb31_t(1693593583),
         bb31_t(759122502),
         bb31_t(1154399525),
         bb31_t(1131812921),
         bb31_t(1080754908),
         bb31_t(53582651),
         bb31_t(893583089),
         bb31_t(6411452),
         bb31_t(1115338635),
         bb31_t(580640471),
         bb31_t(1264354339),
         bb31_t(842931656),
         bb31_t(548879852),
         bb31_t(1595288793),
         bb31_t(1562381995),
         bb31_t(81826002)},
        {bb31_t(262554421),
         bb31_t(1563933798),
         bb31_t(1440025885),
         bb31_t(184445025),
         bb31_t(585385439),
         bb31_t(1396647410),
         bb31_t(1575877922),
         bb31_t(1290587480),
         bb31_t(137125468),
         bb31_t(765010148),
         bb31_t(633675867),
         bb31_t(24537442),
         bb31_t(560123907),
         bb31_t(1895729703),
         bb31_t(541515871),
         bb31_t(1783382863)},
        {bb31_t(628590563),
         bb31_t(1022477421),
         bb31_t(1659530405),
         bb31_t(245668751),
         bb31_t(12194511),
         bb31_t(201609705),
         bb31_t(286217151),
         bb31_t(66943721),
         bb31_t(506306261),
         bb31_t(1067433949),
         bb31_t(748735911),
         bb31_t(1244250808),
         bb31_t(606038199),
         bb31_t(1169474910),
         bb31_t(73007766),
         bb31_t(558938232)},
        {bb31_t(130182653),
         bb31_t(742680828),
         bb31_t(542600513),
         bb31_t(802837101),
         bb31_t(1931786340),
         bb31_t(31204919),
         bb31_t(1709908013),
         bb31_t(925103122),
         bb31_t(1627133772),
         bb31_t(1374470239),
         bb31_t(177883755),
         bb31_t(624229761),
         bb31_t(209862198),
         bb31_t(276092925),
         bb31_t(1820102609),
         bb31_t(974546524)},
        {bb31_t(1293393192),
         bb31_t(221548340),
         bb31_t(1188782305),
         bb31_t(223782844),
         bb31_t(235714646),
         bb31_t(296520220),
         bb31_t(10135706),
         bb31_t(1265611492),
         bb31_t(8872228),
         bb31_t(575851471),
         bb31_t(1612560780),
         bb31_t(1913391015),
         bb31_t(1305283056),
         bb31_t(578597757),
         bb31_t(188109355),
         bb31_t(191192067)},
        {bb31_t(1564209905),
         bb31_t(140931974),
         bb31_t(446421108),
         bb31_t(857368568),
         bb31_t(1375012945),
         bb31_t(1529454825),
         bb31_t(306140690),
         bb31_t(842312378),
         bb31_t(1246997295),
         bb31_t(1011032842),
         bb31_t(1915270363),
         bb31_t(1218245412),
         bb31_t(466048099),
         bb31_t(976561834),
         bb31_t(814378556),
         bb31_t(13244079)},
        {bb31_t(1165280628),
         bb31_t(1203983801),
         bb31_t(1801474112),
         bb31_t(1919627044),
         bb31_t(600240215),
         bb31_t(773269071),
         bb31_t(486685186),
         bb31_t(227516968),
         bb31_t(1415023565),
         bb31_t(502840102),
         bb31_t(199116516),
         bb31_t(510217063),
         bb31_t(166444818),
         bb31_t(1430745893),
         bb31_t(1376516190),
         bb31_t(1775891321)}
    };

    __constant__ bb31_t MAT_INTERNAL_DIAG_M1[WIDTH] = {
        bb31_t(2013265919),
        bb31_t(1),
        bb31_t(2),
        bb31_t(4),
        bb31_t(8),
        bb31_t(16),
        bb31_t(32),
        bb31_t(64),
        bb31_t(128),
        bb31_t(256),
        bb31_t(512),
        bb31_t(1024),
        bb31_t(2048),
        bb31_t(4096),
        bb31_t(8192),
        bb31_t(32768)
    };

    __constant__ bb31_t MONTY_INVERSE = bb31_t(943718400);
}  // namespace constants

class BabyBear16 {
  public:
    using F = bb31_t;

    static constexpr int DIGEST_WIDTH = constants::DIGEST_WIDTH;
    static constexpr int RATE = constants::RATE;
    static constexpr int WIDTH = constants::WIDTH;
    static constexpr int ROUNDS_F = constants::ROUNDS_F;
    static constexpr int ROUNDS_P = constants::ROUNDS_P;

    __device__ static constexpr const int getD() {
        return constants::D;
    }

    __device__ static constexpr const F* getInternalRoundConstants() {
        return constants::INTERNAL_ROUND_CONSTANTS;
    }

    __device__ static constexpr const F (*getExternalRoundConstants())[WIDTH] {
        return constants::EXTERNAL_ROUND_CONSTANTS;
    }

    __device__ static constexpr const F* getMatInternalDiagM1() {
        return constants::MAT_INTERNAL_DIAG_M1;
    }

    __device__ static constexpr const F& getMontyInverse() {
        return constants::MONTY_INVERSE;
    }
};

}  // namespace poseidon2_bb31_16
