import numpy as np
import matplotlib.pyplot as plt
import math
import random


def main():
	classification_cross_validation()


def classification_cross_validation():
	results = [(0, 1.088673959234464, 1.090798662036577, 1.1013328313827515, 1.056554979965335, 1.1179475902634508, 1.0941401154411083), (1, 0.9912667689401574, 0.9951240973813194, 1.0617212851842244, 0.9419496125373684, 1.0269759604159523, 1.00692487249569), (2, 0.7725709337178186, 0.769845375792511, 1.0683276196320852, 0.5230617156658794, 0.8895704851431006, 0.8652306615698094), (3, 0.7127382731117899, 0.7465265958756565, 0.9348984658718109, 0.5392265063262729, 0.8351931398844018, 0.8527141763847701), (4, 0.6644860478165272, 0.721329051183938, 0.8269358952840169, 0.5182693117467799, 0.7854570312535062, 0.8689594620040485), (5, 0.6268567130288066, 0.6654252757839927, 0.765526310602824, 0.37511679092787586, 0.8124134393737596, 0.8075245551917017), (6, 0.6138755108303905, 0.7102953002666215, 0.8611109972000122, 0.40102685837902263, 0.843488311066347, 0.8873796368740043), (7, 0.6709951709596085, 0.735492864123777, 0.9239765405654907, 0.3722124480112986, 0.9204107625957798, 0.902269613377902), (8, 0.704733027800944, 0.8763765142715129, 1.0950908064842224, 0.4216637282408427, 1.0893376815844984, 1.1134329097611564), (9, 0.5727013356014894, 0.6930331850090462, 0.9568375905354818, 0.3766583979931105, 0.8053707101327532, 0.8906123127256121), (10, 0.5986687158657831, 0.755870502596112, 1.1902760068575542, 0.3309666524270471, 0.9314846056787407, 0.9746324325702629), (11, 0.5296523707327703, 0.6246356836648146, 0.7695974151293437, 0.32875683520782695, 0.7647069000145968, 0.7764020930139386), (12, 0.555456580490419, 0.6748389078976107, 0.7679120471080144, 0.3475933411491992, 0.8017100910930073, 0.8919142256585919), (13, 0.5143216240618793, 0.6482451213001581, 1.0316938519477845, 0.25808295543594995, 0.8110239425783649, 0.8493689333601874), (14, 0.6233352590479134, 0.7780516760657444, 1.2712135672569276, 0.43994390736340283, 0.8609497806824306, 1.0084351167386891), (15, 0.5199385263436092, 0.6652859461942734, 1.0537172118822733, 0.24738815381164217, 0.8046038417036042, 0.9327331696237836), (16, 0.5069371065468838, 0.66017267169523, 0.7188705476621787, 0.3476296929491944, 0.7619659554569379, 0.8990082704291051), (17, 0.6108854839587495, 0.7466934996628422, 0.8095973869164784, 0.3908601349659278, 0.9243271961698637, 0.9335295145532915), (18, 0.4771748036189271, 0.6295977723513853, 0.8405020515124003, 0.2590875993155184, 0.7144822788643924, 0.94076495640436), (19, 0.4359159029056097, 0.5783037608148303, 0.7920798559983572, 0.3236390488391126, 0.6071830050581518, 0.822537191592309), (20, 0.4513221669017345, 0.596959659049582, 0.9029852822422981, 0.25954316187909, 0.6308036846903098, 0.9232008919043808), (21, 0.401722580142522, 0.5366158678993919, 0.634211469690005, 0.23814170730476547, 0.569610008281683, 0.8474598523248367), (22, 0.41097066375877433, 0.564952894328013, 0.6893595457077026, 0.251775581910829, 0.6207737725858083, 0.8583191484212875), (23, 0.42930277605678496, 0.5750275147463664, 0.872408123811086, 0.23813793804430164, 0.6136681000733755, 0.8952795516566506), (24, 0.3796267268210488, 0.5336838641174657, 0.7383627931276957, 0.278712273022171, 0.5436533925719286, 0.8059338060426241), (25, 0.38125684694701534, 0.5718266520234029, 0.7873543617626031, 0.23543806114990923, 0.6665781387418974, 0.8261153280107799), (26, 0.34958049059340984, 0.5604487995142133, 0.8434400975704193, 0.24797473197339331, 0.6181952316742622, 0.8259942280384), (27, 0.35675065031434855, 0.5295420102627998, 0.8117388832072417, 0.1918223595093246, 0.6108117984618088, 0.7939925521270049), (28, 0.3499896711189511, 0.5363211486816146, 0.8465345899264017, 0.2448441204816188, 0.5942872568630264, 0.7712558946923866), (29, 0.3374782406875564, 0.5237136059803286, 0.7874509344498316, 0.24592274223931368, 0.5165692064637208, 0.8390816037723681), (30, 0.3770301638208968, 0.658375991149549, 0.8281127691268921, 0.16698734119551914, 0.8219513763324358, 1.0171220242093337), (31, 0.33963307636710355, 0.5597748596155729, 0.9968340923388799, 0.2829785263560953, 0.5999530211556703, 0.7817037799967304), (32, 0.3147160447633182, 0.5510341845557242, 0.7690302570660908, 0.31014296139034025, 0.5685484976052334, 0.7932468535739701), (33, 0.31570625931205154, 0.5652710732861771, 0.9110649106403191, 0.281118492096292, 0.6058011713840268, 0.8098385631182821), (34, 0.3264336361937329, 0.6422549008721199, 1.2736172763009865, 0.22560230536414388, 0.695479224645958, 0.9904454415588347), (35, 0.29491776943961434, 0.5637484699227517, 0.9992879165957371, 0.18738536184421528, 0.6232764099704047, 0.8830077315452604), (36, 0.2845878915700191, 0.5482854467421722, 0.9209780381526798, 0.25350903462225116, 0.5866036379987717, 0.8050309507888581), (37, 0.3000353156319078, 0.5451184184908323, 0.732241208354632, 0.2634750592131345, 0.5823489073748479, 0.8154276174265055), (38, 0.2927805806788138, 0.6041967180042487, 0.8859199337661267, 0.2585288923267206, 0.7534329515652002, 0.7842936438403796), (39, 0.2791206425233805, 0.5623839859248286, 1.0996900742252669, 0.3149063460997881, 0.6000170527769609, 0.7360021863560363), (40, 0.2747326824062321, 0.5468895995801519, 0.8482727212210496, 0.2594922061828617, 0.6284507322941542, 0.7453530804041478), (41, 0.26714444454279773, 0.5633311008409622, 0.7452466050162911, 0.2731777738120847, 0.6829539832038549, 0.7306909103668771), (42, 0.25902622529332703, 0.5214557824147517, 0.7331951794990649, 0.3282586552515996, 0.5381941736238846, 0.7063286125494229), (43, 0.2926350186200591, 0.649330030511747, 0.7889955112206128, 0.23441037106998514, 0.7314697321440048, 1.0304955082594498), (44, 0.2807380147897878, 0.6119989919282977, 1.0688019569342335, 0.2198763045413813, 0.7379621919963564, 0.855426830467673), (45, 0.23797457948006956, 0.6058736575518806, 1.2025009249647458, 0.22119061829326883, 0.6266554370068688, 0.9646042674751382), (46, 0.24630321761777388, 0.637046408416474, 1.0120054706931114, 0.20523240397114179, 0.7515210790356406, 0.9583562798855995), (47, 0.27498404951942174, 0.7074026853242886, 1.4178582049906254, 0.20171182077593539, 0.796098727758891, 1.1051046333732488), (48, 0.22636320101273089, 0.6238553722200001, 0.7111637890338898, 0.36111790635623486, 0.6467722337380915, 0.9057701825945979), (49, 0.2666457416573484, 0.7599201889534515, 1.4153854972372453, 0.15568377938012645, 0.9759946311477613, 1.1119481298234781), (50, 0.24080350361345948, 0.6379793674108547, 0.6405658091108004, 0.18738832376236442, 0.768751679754229, 1.0170434104814432), (51, 0.2405313760747899, 0.5968763419026265, 0.7028589663406213, 0.25362060816345683, 0.6808044561431902, 0.8915012052994844), (52, 0.2361177507908299, 0.6923147064782561, 1.0769925504767646, 0.2103631830523706, 0.866439380951279, 0.9917735071889628), (53, 0.21794169037007866, 0.6483431434152268, 1.0626519887708128, 0.18489905656839783, 0.8980125671300394, 0.8153891055556841), (54, 0.25109405012171043, 0.9296332905270881, 1.5112256539054214, 0.2065027942587367, 1.3452300805428394, 1.1640911645262888), (55, 0.23353866926476066, 0.5938010964171195, 1.030892970599234, 0.26812178731495667, 0.7465172803568307, 0.7204041227633585), (56, 0.22188182078653104, 0.7045733389801395, 0.968798736234506, 0.27646083380664566, 0.7938118743758397, 1.073245827332543), (57, 0.19878094716727882, 0.6788578205209197, 0.975201715528965, 0.2750267336182507, 0.7067523498472459, 1.0975167816293332), (58, 0.23279854109217174, 0.7536933107163218, 0.6798739222809672, 0.2647557384178123, 0.8325390047714198, 1.2642510684250479), (59, 0.22201834989092578, 0.755773537387134, 0.9670025942847132, 0.30538957313554077, 0.9561808284522345, 1.006008396574655), (60, 0.19908682853246226, 0.6413932445860527, 1.3691703112485507, 0.20083037121144892, 0.7578242588040834, 0.9168768835463791), (61, 0.2040956773427901, 0.6527562593418318, 0.9351538630202413, 0.2549933933968599, 0.7260092799482369, 1.003049063297335), (62, 0.18239599448773056, 0.676624206110879, 0.718840777579074, 0.14680326420169412, 0.8129129989050758, 1.1406000277525814), (63, 0.23108511188713984, 0.7191131159345289, 0.763015249123176, 0.26780083537624083, 0.8861900173644331, 1.0423693162999825), (64, 0.2593728777396991, 0.8119531075502697, 1.0821994135078663, 0.14274056626486056, 0.9851985348687999, 1.3632679782863493), (65, 0.24280036862461604, 0.8907514575682526, 1.2851000170456246, 0.32263757311578684, 1.056156200591371, 1.3080945234855599), (66, 0.17691888403654035, 0.7004278852679785, 0.7653304812571756, 0.2561942989602865, 0.8717953592464089, 1.0057032131111814), (67, 0.16222572071763242, 0.6738493343967168, 1.1796647662607331, 0.1954586301347555, 0.7933131139119015, 1.0261897469266545), (68, 0.16706074654342667, 0.7273026201833522, 0.9819707348942757, 0.2510074266752587, 0.9057967560522516, 1.0335555733379769), (69, 0.19219389222343336, 0.6334310120167697, 1.326864191517234, 0.4151776959033303, 0.6660945834223171, 0.7536673504063429), (70, 0.1508624495644401, 0.7525439271215929, 0.8220017545313264, 0.2753751484952956, 0.9708443916272111, 1.0329914823103499), (71, 0.1791915844635907, 0.7964311732145111, 1.0790108820733926, 0.2187733242471549, 0.9888516151250997, 1.2052718042481252), (72, 0.1792453785104667, 0.792177489172491, 1.406374292979793, 0.24127980864748325, 0.845576943655026, 1.3098738071042932), (73, 0.1864916100555929, 0.6737783356275379, 1.8183492006035522, 0.37881544496689284, 0.7601969214534987, 0.7458597847461885), (74, 0.2003825592449638, 0.8339232582828919, 1.256611517506341, 0.3219384413740219, 0.9726607226005863, 1.2140613055004827), (75, 0.17293161477918023, 0.8749981021145957, 1.8718089903947355, 0.281482321194489, 0.8798154543832707, 1.4546062861912392), (76, 0.18100601669517946, 0.8066890937464019, 1.3340618932192834, 0.1689752776917231, 1.029907179158494, 1.2100853585477591), (77, 0.2622586331861835, 0.9658244729198067, 0.7037916122494304, 0.1756494608152684, 1.3406100891083175, 1.4695081523603353), (78, 0.2019784066071295, 0.6689625129124932, 0.9260358645309074, 0.1746516875448668, 0.8398949893228727, 1.0077686723896626), (79, 0.1386000091331085, 0.7473459387598762, 1.1335352108969043, 0.28321377461162134, 0.9320990390683017, 1.0096404622505473), (80, 0.12607373279479392, 0.7421080234378474, 1.2490435096260626, 0.34997622438887144, 0.8217066635524726, 1.0422165146170597), (81, 0.13729181018654749, 0.7584051098704905, 1.0229996773580448, 0.2450313612214503, 0.947760984887331, 1.0942242917131362), (82, 0.131952617539569, 0.8285569200785569, 0.7911924132689213, 0.247917354708026, 1.0617133000280168, 1.233549031346136), (83, 0.14177249781863477, 0.7349519155143387, 1.018967412536343, 0.3530441282151696, 0.8314893391475359, 1.0329460785863152), (84, 0.16049479066299932, 0.5876020921725069, 0.8539432093501091, 0.3374247608898003, 0.5686245221143454, 0.8846170634657325), (85, 0.23933017146917057, 0.727654969127362, 0.8894803007443746, 0.3236475009715491, 0.7987125254373549, 1.1072233720109839), (86, 0.11955348468057046, 0.7037507159290991, 1.0682491350919008, 0.2988881415993135, 0.8075605321595023, 1.0079101319847816), (87, 0.11878636785821041, 0.8526330623254685, 1.0976023644014883, 0.2681526664841385, 1.134121025714697, 1.1521198555965593), (88, 0.11465369054965992, 0.7477752338871548, 0.6569210081671675, 0.21771392383969176, 0.9556697529168524, 1.133047056576455), (89, 0.13259967682378582, 0.7949367566400515, 1.4646622312022373, 0.281938906380281, 0.954274085092332, 1.109936541780348), (90, 0.1517911597994484, 0.8737750248042219, 1.203695482139786, 0.21365146703730717, 1.1486319865675387, 1.2636293144748243), (91, 0.11041644343979097, 0.8281747136299608, 1.1028339871981492, 0.33849438347389627, 1.0037129433508172, 1.152132998665477), (92, 0.1101994342664977, 0.8030268748197787, 1.5334196651354433, 0.3101161604450248, 0.9081411844864613, 1.1589826828230056), (93, 0.11568213356834973, 0.7644551715767696, 1.2801413658385477, 0.28433578814665544, 0.8302676642455337, 1.1918916293481503), (94, 0.14019585413024832, 0.7568533468283645, 0.8829031757079064, 0.23120896535645172, 0.9412556398847426, 1.1360284004070667), (95, 0.13452600956583055, 0.8754755564300448, 0.9373421642987523, 0.30134550929495146, 1.1767746209301084, 1.1626101876590467), (96, 0.1450158564409514, 0.777275517394753, 0.6658811677237585, 0.4031156418317099, 0.828147717233908, 1.189518383574662), (97, 0.10606239988691234, 0.8607895472315431, 0.7018720634281636, 0.42761583849245827, 0.8487703496930619, 1.4410504205830987), (98, 0.13149544526431198, 0.7290300718050575, 1.382024928772201, 0.2719157993976281, 0.8558828781996494, 1.022101874963575), (99, 0.11342269618482446, 0.8491296819626777, 2.0523092396528226, 0.3185384572154018, 0.9434988581883855, 1.1945401726475675), (100, 0.11257018048352573, 0.925862854147614, 1.4955698583895962, 0.28480950260462934, 1.162518786189211, 1.3082883344226874), (101, 0.11073591691086149, 0.6212393975813142, 1.4895419651331774, 0.2537529951977721, 0.6830407131546409, 0.860053924719495), (102, 0.12826679666586918, 0.7396844191294694, 1.074575005704052, 0.21540605959404957, 1.021100916154965, 0.9505615361624805), (103, 0.16725716652606393, 0.8249681033164016, 0.8760501643021902, 0.31541009428157085, 0.8932085553752946, 1.3567961105353104), (104, 0.1346070445077599, 0.7368130907139685, 1.1739124670624732, 0.22825767783817136, 0.9251404910377031, 1.0416574099138842), (105, 0.19787633972720214, 0.6833835188243284, 1.3034111731384959, 0.42565353663130906, 0.7554026664032747, 0.8093839161926869), (106, 0.0969048835683997, 0.811868597920358, 1.4708246720333895, 0.2856594522407956, 0.9815175417165463, 1.130653478582904), (107, 0.08189965609678766, 0.8336227662978628, 1.0065208461135626, 0.2532040949029116, 0.90151625970071, 1.4355012521158357), (108, 0.1459592252083941, 0.8651214843529483, 1.5973037799137577, 0.18468139554949053, 1.2165467303313615, 1.1124390654503515), (109, 0.0920521381348956, 0.8035113582107435, 0.9868736338491241, 0.36391036381851244, 0.9338937187217264, 1.141765298535057), (110, 0.08436465887586234, 0.7442012747473308, 1.0158010865872105, 0.34089394344455826, 0.7976241402731562, 1.1305684944394512), (111, 0.09997787166265311, 0.6834372067000044, 0.7961115550783385, 0.366047864713372, 0.619576666862815, 1.1499312670525748), (112, 0.12150987925579189, 0.958292713022372, 1.083344039817651, 0.24552404907348463, 1.1223835664484847, 1.5987585398987363), (113, 0.12065305896013345, 0.8826549235756368, 1.1719782312284224, 0.3196491524640167, 1.0302937539960872, 1.33436893912303), (114, 0.12904599327880178, 0.8098025867552008, 1.253428835561499, 0.19638138348653533, 0.9337631856706807, 1.3335204603692559), (115, 0.12965109248371642, 0.7825050354456464, 0.8951945936462532, 0.21157178202040328, 0.9220378928641454, 1.2823728409145718), (116, 0.09643940003391453, 0.8507714164852768, 1.173267283154928, 0.2342570645947315, 0.9947704012159638, 1.3690719369671978), (117, 0.09329660518691843, 0.7160560225878324, 0.955605159123661, 0.3144254371993162, 0.7438781307416107, 1.1407691619801557), (118, 0.1623698081764938, 0.7292114479261854, 0.6752094413464268, 0.2237217555932907, 0.8883678452695245, 1.1458900451161382), (119, 0.08079173482705988, 0.9110012662789452, 1.1791682166435444, 0.4736632743692189, 0.9951855003898382, 1.2975693776915584), (120, 0.08956308664632379, 0.7894045128117817, 1.4593748069368302, 0.3063081372517151, 0.9653539726984327, 1.044088970585943), (121, 0.06520276600320846, 0.8606456765919515, 1.928770642541349, 0.3975680420982074, 0.9142096449116884, 1.1993078931992127), (122, 0.096843515726891, 0.6968782939157433, 1.1463391257605204, 0.20470392679490543, 0.8695766576504679, 1.0011270984376617), (123, 0.09560020408479464, 0.834497936371678, 1.066300017107278, 0.387253657186187, 0.8885730035594527, 1.2807485660865607), (124, 0.1014270222920229, 0.715092385357558, 1.434467046521604, 0.2878946471436549, 0.7837540528598097, 1.0415160319323464), (125, 0.08702855919721743, 0.8234196520911807, 1.523383188759908, 0.3693407767260342, 0.797802837333124, 1.3171141066994732), (126, 0.08094972392366678, 0.9475543955125976, 1.2226110492600128, 0.5460140182037823, 0.9972527888341811, 1.3363614846322798), (127, 0.08448817471883689, 0.7440468167921691, 1.3880901226463418, 0.4584490743974734, 0.791069505113447, 0.9357528307987659), (128, 0.07403971715513616, 0.7304828392737494, 1.6062874950817787, 0.31802160265296736, 0.8340766216328578, 0.9661404579449352), (129, 0.09593729586061037, 0.8883391727806978, 0.839089361828519, 0.2276868520529725, 1.0636373941557726, 1.4750511338628913), (130, 0.0824636639620934, 0.8211045588335295, 1.9030270965184173, 0.3314545601409395, 0.9531947930464485, 1.0817595580403219), (131, 0.07173693251869803, 0.8985523106340463, 1.5338484025744643, 0.5185056483589683, 0.8594121720119278, 1.3287493542163045), (132, 0.11685186409068885, 0.7412408683422391, 1.3783736307096357, 0.36833056718403084, 0.863644139312836, 0.9380900566462116), (133, 0.13096596808078634, 1.0429346394582844, 1.8321946548175523, 0.2505765135637459, 1.2919879529381095, 1.5629095424734734), (134, 0.13690077319382726, 0.8442235325929984, 1.948623398815592, 0.27985345861824823, 0.994262556397027, 1.1695484692688427), (135, 0.08529899056516889, 0.8410229808557648, 1.5635102106025442, 0.3704212192293277, 0.8951023152127968, 1.241240501260533), (136, 0.06678354873211148, 0.7864807156582905, 1.1360338340202967, 0.3044048777716044, 0.9857982524178145, 1.0565089855302325), (137, 0.05974354093794803, 0.7991806221928602, 1.9748785607671986, 0.3795586055860361, 0.9079994756724666, 0.9905995290723602), (138, 0.09415807628350997, 0.8583220020103317, 0.940513503101344, 0.3972849090318489, 0.8579604922941291, 1.4201874933206704), (139, 0.09761635066911153, 0.7119410140773528, 1.1128627430725222, 0.31597700157428454, 0.8922926414223841, 0.8932263433786032), (140, 0.059774244983489024, 0.8388063978561016, 1.5230747818015515, 0.2577645068032084, 0.9700702656730184, 1.2752472032251885), (141, 0.08682113721169646, 0.9306208320106195, 0.9858750923074695, 0.2460422061113425, 1.2375780542800172, 1.3484126302414456), (142, 0.12668556903562578, 1.0189140155665706, 1.2863271777789729, 0.37002169442986454, 1.1001193588377973, 1.6730950263890731), (143, 0.11725983443830931, 0.7352338486857012, 1.2089514940200994, 0.28534543158720516, 0.8883841589911566, 1.010256093629156), (144, 0.08152073520632792, 0.8106638904343111, 1.8966423297611377, 0.26567409130722774, 1.049455979895661, 0.9915144489736799), (145, 0.09892087499921591, 0.963757875684969, 1.1100158229470254, 0.4203979460341223, 1.1121388898852942, 1.4118826132077058), (146, 0.08030075707761564, 0.8557853663816156, 1.4583986667916178, 0.25770728672145377, 1.144943084858896, 1.106814718417122), (147, 0.11736032407917475, 0.8583601371849681, 1.2408845195313234, 0.22286717277196172, 0.9483334546397356, 1.4660734509537745), (148, 0.06651829397463686, 0.7954515897645514, 1.3816525015359125, 0.3739699159696251, 0.9247119531530492, 1.051046907168377), (149, 0.06566987175090361, 0.7627993230326678, 0.9975459166492026, 0.3981382944015147, 0.8329295789243494, 1.083510871884545), (150, 0.06416297347842363, 0.8635018196671441, 0.6988011772045866, 0.35702753300832996, 1.0347318960539638, 1.281594495388636), (151, 0.08675510553847574, 0.806939128545823, 0.871530629034775, 0.37848056820278264, 0.9108240638747234, 1.1862730127273144), (152, 0.07212314970922365, 0.8676508975014424, 1.6602772225737377, 0.3810713680212764, 0.974153542434224, 1.2042728159671316), (153, 0.058116298429191325, 0.8853880190032224, 1.6943961907954266, 0.373631056673347, 0.9344403989061995, 1.3305727451313534), (154, 0.08345383531116046, 0.9329533184211002, 1.9943268270930274, 0.37355972675184873, 0.9116106269756711, 1.4965045591368968), (155, 0.0583988190100067, 0.8743638051140477, 1.5532724798150108, 0.29380672496924365, 1.043577350529409, 1.2583565753248889), (156, 0.11299194682454018, 0.8448077200073463, 0.9750861708540469, 0.3811906056300864, 0.7948593421436954, 1.4713392565459353), (157, 0.0931620821287955, 0.8763898742069594, 1.378348782923907, 0.36996885540848523, 1.0564134051216256, 1.1801733482708878), (158, 0.0713354749490369, 0.87505461982399, 1.4346656599353689, 0.2399122848389906, 1.0959613417526455, 1.2735228349689092), (159, 0.11179052964774643, 0.7861130555327521, 1.3837959923936676, 0.47489823753082133, 0.9306505071325287, 0.8814790792649176), (160, 0.17370292906783813, 0.7956506536318804, 1.1658401990309357, 0.17663089783176442, 0.9875964012214808, 1.2432317286772132), (161, 0.12522079008233464, 0.8819534053955016, 1.0981466906766097, 0.21020001442501274, 1.1385510941269525, 1.3290322702088249), (162, 0.08253228240907964, 0.8530447847939047, 1.8592126491479575, 0.3732325802296962, 1.0221801164649398, 1.0616383142251482), (163, 0.06572238115478078, 0.8416441719733606, 1.2865142552143274, 0.3307123999386642, 0.9737168066013963, 1.2263252600366834), (164, 0.06465521035199903, 0.8128690261268045, 1.5642189519113647, 0.3716480693842723, 0.9522041581928579, 1.0537785146455518), (165, 0.0906265812609169, 0.8533617173168078, 1.3810373213142157, 0.3518708174468956, 0.8535294440649525, 1.3966672773823228), (166, 0.06019982472289735, 1.0751742833665399, 1.507449116006804, 0.24480562436876957, 1.1967200928799722, 1.8740581998389607), (167, 0.07360054362941391, 0.774641395184293, 1.292059836179639, 0.26997577122562316, 1.036587896610834, 0.9601843065518235), (168, 0.06689055088220446, 0.6790713955539536, 1.1972696124808864, 0.2889561449891266, 0.8328730528453083, 0.871969986608858), (169, 0.07300431406919207, 0.7593577535277132, 0.6921924734061273, 0.28268873054690713, 0.9192407698363486, 1.1411639758492207), (170, 0.052296017657088224, 0.9802990569294632, 1.4442438736851424, 0.515216031834292, 1.083105110434739, 1.3455983787004642), (171, 0.06533424201355167, 0.9455097470501892, 1.40922497313004, 0.5227033919729528, 1.060987315571801, 1.2406291104102496), (172, 0.09002585845940063, 0.7656954682187694, 0.9758419682582219, 0.35941522868242254, 0.8816007253854437, 1.0784595166695332), (173, 0.12825076524738294, 0.953912911124579, 1.927545656512181, 0.19368068818921028, 1.1245903665028838, 1.5144405242455128), (174, 0.07803428959188405, 0.8445837624411152, 1.0399942689885695, 0.387316391229864, 1.0459824989193587, 1.1044330841607044), (175, 0.05874354864993453, 0.7569689887645643, 1.4050604939423161, 0.25124149294890485, 0.8619387376732971, 1.1416781322365694), (176, 0.07114485718734986, 0.8336173075916185, 1.0290236551625034, 0.1886360407025093, 1.0399126401748315, 1.3203566353017313), (177, 0.08683892134004176, 0.8320976640918467, 1.319140257548952, 0.33308500384045614, 0.9462399167930338, 1.220368677167884), (178, 0.055031488221416826, 0.8834289463306478, 1.6660716291392115, 0.3701577659226677, 0.9574673434392352, 1.2998597806253125), (179, 0.11527579456555677, 0.6929986201792615, 1.1901381307281553, 0.3677341673839354, 0.8288008660129516, 0.8333666319079811), (180, 0.07078363614269093, 1.0067605301636788, 1.3264153794717761, 0.3904029547340372, 1.088846438211583, 1.611220917595105), (181, 0.07521262530890759, 0.8113006460297614, 0.6558614895116383, 0.1894758522921213, 1.0820132108951208, 1.233518027275425), (182, 0.05744518059789988, 1.0995555580506748, 1.4564476868060108, 0.3755576316507301, 1.2390883979117324, 1.752595342421689), (183, 0.09674455433204146, 0.8511887125405883, 1.1112181931190814, 0.32348874341480466, 1.0485688919359162, 1.1944055454681766), (184, 0.045049802566607774, 0.8732954294843491, 1.0748091421594532, 0.5816642427708649, 0.9621147602345412, 1.0822430836363122), (185, 0.07043133308664906, 0.8067129274225849, 0.5901401891867863, 0.3723308397567874, 0.8701982430626809, 1.2925211196849444), (186, 0.056859357009236125, 0.7320527731740982, 1.1978494181705173, 0.2456424173136629, 0.8999020003552954, 1.033355006269853), (187, 0.08875464383510104, 0.9880911226070899, 1.5555152522594047, 0.25708026383361093, 1.1062375875759267, 1.647316057197068), (188, 0.05091405667075802, 0.9223869578668837, 1.2601890537383345, 0.25964724915932375, 1.1243428487942633, 1.4154605482559344), (189, 0.05531931393399106, 0.8743289217547678, 0.9967968512675724, 0.4470188040200448, 1.0012332126519121, 1.2114292468459156), (190, 0.07308345507170197, 0.9436786830286212, 1.648076318469733, 0.2792424675960513, 1.0446064541264135, 1.5229549777414484), (191, 0.08747426453678656, 0.7709130428157734, 0.7059057749807834, 0.2974547143182276, 0.8845423646513566, 1.2125807706176732), (192, 0.06811920362465121, 0.9363899597465338, 1.8905186292807532, 0.6137304992861583, 0.8094273635119703, 1.3682209517980966), (193, 0.09625924415234188, 0.7989721666157293, 1.1837363321644565, 0.18953586560680621, 0.8893343572111088, 1.3733651086451797), (194, 0.04565943384907698, 0.8070106139958954, 1.08275679400734, 0.341629420955694, 0.9306432085756929, 1.1725848995865122), (195, 0.12171814617878536, 0.9537555331438072, 0.7634967906487873, 0.21718245386468876, 1.209249966268392, 1.5452713893125236), (196, 0.04511217664006232, 0.7840113141539409, 1.5841253506485373, 0.255122280738906, 0.9381712497263601, 1.106021521208699), (197, 0.08516089639104854, 0.86570144255399, 1.1531745406100526, 0.29248527393381313, 0.996798538295001, 1.3533654331666751), (198, 0.08341802271183096, 0.9023038116652857, 1.130778575859343, 0.22453822450986421, 1.1525681089677888, 1.363776625307226), (199, 0.07100461383051788, 0.9287612740096439, 1.3657080839155242, 0.41635959629330255, 1.1205205025123754, 1.2336546154018269), (200, 0.06664853933889923, 0.8034073612090833, 1.080909931814919, 0.2979195298099851, 1.0796871981899179, 1.006804494292848), (201, 0.0638457943644629, 0.8459093921825315, 1.1805422178062144, 0.407499059730686, 0.921584363904085, 1.2354478821890476), (202, 0.08541801185847711, 0.9242459691460523, 1.4851986212267851, 0.25611678658297565, 1.1051264057727375, 1.419120776393554), (203, 0.06160673576562267, 0.7682503925476931, 1.1049068248520295, 0.2323141292936631, 1.0566889793034107, 0.9836251867686016), (204, 0.08384534932021329, 0.8145960620605085, 1.46308481975672, 0.18937425128554305, 1.1645562879690492, 1.008015845994601)]

	values = [[],[],[],[],[],[],[]]
	titles = ['Epochs', 'Training', 'Testing', 'Testing BCIHM', 'Testing AISD', 'Testing RSNA', 'Testing CQ500']

	for result in results:
		for i in range(7):
			values[i].append(result[i])

	for i in range(1, 7):
		plt.plot(values[0], values[i], label=titles[i])

	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	# plt.savefig('training.png', bbox_inches='tight')
	plt.show()


if __name__ == "__main__":
	main()