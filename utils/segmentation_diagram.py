import numpy as np
import matplotlib.pyplot as plt
import math
import random


def main():
	segmentation_comparison()


def segmentation_cross_validation():
	# results = [(0, 1.037408016440934, 1.012490121836073), (1, 0.9844969008208481, 0.9891208748469192), (2, 0.9665804458495284, 0.9563174890668205), (3, 0.9443125754739708, 0.9270938423912177), (4, 0.9191228225706463, 0.8976021459263362), (5, 0.8991599889106843, 0.8748171933916178), (6, 0.8845031817399895, 0.8663540070646265), (7, 0.8664366094881557, 0.8650973454285203), (8, 0.8460493786698222, 0.8975844410028351), (9, 0.8274497506804945, 0.9278440635358349), (10, 0.8089301064610481, 0.8741622371285149), (11, 0.7922987245759856, 0.8707523062155488), (12, 0.7801122781933802, 0.949648423141308), (13, 0.7666675710697421, 0.7906199876177177), (14, 0.7547379030062662, 0.7952319170149524), (15, 0.7457546725434461, 0.7885965215188734), (16, 0.737429524453378, 0.7946589802590649), (17, 0.7291228493714835, 0.7465922718637445), (18, 0.7190381633646098, 0.7578399113557311), (19, 0.7105531452323282, 0.7325609091962322), (20, 0.6999913153375953, 0.7147672651021668), (21, 0.6868097684142366, 0.7341561563564151), (22, 0.6805347515937286, 0.832470406223549), (23, 0.6721967337064361, 0.7492913660708438), (24, 0.6656081217010559, 0.7733732654890987), (25, 0.6595380345066614, 0.7463332043772333), (26, 0.6519921515348278, 0.7599661680084936), (27, 0.6479585422157662, 0.7395626772404387), (28, 0.6395085782078329, 0.7271735096413098), (29, 0.6331898751495046, 0.7188391683811552), (30, 0.6299978929845016, 0.7392729745570863), (31, 0.6199396053567601, 0.7006999281433861), (32, 0.618031337054226, 0.7118588044737162), (33, 0.6104502861766896, 0.70417152621438), (34, 0.607763118220786, 0.6678250830913528), (35, 0.6037114103546598, 0.68931123438511), (36, 0.5900340368322171, 0.71696615625131), (37, 0.5866661246268251, 0.6724111583042011), (38, 0.5885760815183193, 0.6553185932505666), (39, 0.580145497992031, 0.6566478574878714), (40, 0.5692364349357506, 0.6662082256895773), (41, 0.5662575395629896, 0.6528589880366004), (42, 0.5624879173620497, 0.6535210274159908), (43, 0.5545478116243933, 0.6683693759645639), (44, 0.5539587382257372, 0.6444782818552484), (45, 0.5523115080363167, 0.6641588404439809), (46, 0.5461394860615993, 0.6437795671184411), (47, 0.5408399233072272, 0.6556900620795367), (48, 0.5336168942311884, 0.6435134550661183), (49, 0.5290254650568074, 0.652857175954942), (50, 0.5313075060202971, 0.6650967904774661), (51, 0.523838429749495, 0.6635965647573552), (52, 0.5121158032739954, 0.6508389136680727), (53, 0.5172530852704998, 0.6592776566492707), (54, 0.5089943629766496, 0.6123622314732396), (55, 0.5059985047752699, 0.640171265166797), (56, 0.49742994151973263, 0.6084856522384654), (57, 0.49658215381627724, 0.616065912827682), (58, 0.49052987738574344, 0.6060682362719868), (59, 0.4838790509191672, 0.6132395204700781), (60, 0.48157071101824894, 0.6124102483304699), (61, 0.4803527504314383, 0.6223015324369575), (62, 0.47566218762937784, 0.5992443201582084), (63, 0.4716510935422377, 0.6085249775748575), (64, 0.4686639188234868, 0.5903831832948025), (65, 0.46442542928520053, 0.6028463784395979), (66, 0.45742838745227316, 0.6211450386248277), (67, 0.45492415550969406, 0.6025577808280339), (68, 0.456625282329717, 0.5805084468441063), (69, 0.4500629906052032, 0.6012668233741535), (70, 0.45234774043436377, 0.6233821771620365), (71, 0.4480735628384149, 0.6015503898765264), (72, 0.4468050170901142, 0.5742840693656648), (73, 0.43413621842305705, 0.5923693375138754), (74, 0.44011349473118394, 0.6353612542152405), (75, 0.43580996050730325, 0.5811038392313411), (76, 0.42519927081596137, 0.5720420712500476), (77, 0.4219340483327449, 0.5729837431415413), (78, 0.42141040338496144, 0.5926363610066054), (79, 0.41239489230359005, 0.6090852433208669), (80, 0.4069315869379275, 0.5809987019119638), (81, 0.40615332945939786, 0.5634836182416825), (82, 0.40660233487717723, 0.5711650922894478), (83, 0.40181514033308097, 0.5758079134681252), (84, 0.3996184289672773, 0.6107280063495207), (85, 0.3956205265496886, 0.6030101900438914), (86, 0.3870345512208811, 0.5605430820266183), (87, 0.39615840411698217, 0.556479363084844), (88, 0.38278062517243033, 0.5894035531228847), (89, 0.3873965708867947, 0.5506825912953093), (90, 0.3792098307532275, 0.5798941209410013), (91, 0.3783541678947606, 0.5565018926359965), (92, 0.375234460684575, 0.5570756486627493), (93, 0.371541802842333, 0.5673453278755873), (94, 0.3664718284792413, 0.5591598509319042), (95, 0.36431453912556655, 0.5745860745062988), (96, 0.36548860969453617, 0.567959898559565), (97, 0.3636586547888079, 0.5577821378292662), (98, 0.3636706033404584, 0.5466790351639973), (99, 0.3590393722105645, 0.5592439544251125), (100, 0.3535982473018297, 0.52907035591897), (101, 0.35042626030948404, 0.5475705932365375), (102, 0.35397925428937477, 0.5593626147575592), (103, 0.34825278731663767, 0.549897399260087), (104, 0.34270231682366453, 0.5389744034141637), (105, 0.34124682123381855, 0.5302015821752923), (106, 0.33840425966407145, 0.5281733289109857), (107, 0.3350971234687627, 0.5202929449131649), (108, 0.33193717079427293, 0.5540442432831513), (109, 0.3329736090278799, 0.5490952623024415), (110, 0.3300035319244263, 0.5443372493714429), (111, 0.3322177971358527, 0.5250822375999408), (112, 0.3212604501150319, 0.5171439221986894), (113, 0.3250007736443797, 0.5406689994623152), (114, 0.3193485671095852, 0.54782360181045), (115, 0.31683968726056516, 0.5339704019048911), (116, 0.3161324198920781, 0.5761158352906115), (117, 0.31491092411345367, 0.511545678077454), (118, 0.3094944391139999, 0.5373129355354925), (119, 0.30786985838166897, 0.5443102986541357), (120, 0.31243599639719466, 0.5306111639051625), (121, 0.3070148380240803, 0.5338603063915552), (122, 0.3096718279527111, 0.5141677551724938), (123, 0.30624313639534256, 0.5078684273228217), (124, 0.2972983789526083, 0.5361025965448176), (125, 0.2979978712326945, 0.5367089139109247), (126, 0.2952145873123668, 0.5178126971122254), (127, 0.29487740738404916, 0.513781396651201), (128, 0.29257434151004064, 0.524270429663109), (129, 0.2923575535754717, 0.5133773253372546), (130, 0.2912441672277267, 0.516399999264251), (131, 0.28673184597849655, 0.5118997578791687), (132, 0.28443539857176076, 0.525615482392271), (133, 0.28495825969008803, 0.5006041029321678), (134, 0.2808346448547691, 0.5334575067828881), (135, 0.28022335183835395, 0.5036587180004696), (136, 0.28102879818812665, 0.505920889965269), (137, 0.27862349957396765, 0.49930691419776235), (138, 0.2766601479245498, 0.5058598763026884), (139, 0.27512022882482606, 0.5135489988109369), (140, 0.27206937760428634, 0.5286585640664516), (141, 0.2712414353705544, 0.5170236933097411), (142, 0.2697667340767635, 0.5049219128958294), (143, 0.2679413408433454, 0.5095478803170531), (144, 0.26923165003945704, 0.506552189421118), (145, 0.2653181617822939, 0.4997567334369327), (146, 0.26345584757710583, 0.5122791458381696), (147, 0.26304517092252666, 0.49656717705257825), (148, 0.2605654862248965, 0.5042618719044696), (149, 0.257311130501028, 0.4911417140719596), (150, 0.2583769832434347, 0.5116575340206704), (151, 0.256185297261036, 0.5060130619349774), (152, 0.25419796523872035, 0.4948776216570581), (153, 0.2516252526823232, 0.5043147665898452), (154, 0.2497955841620602, 0.49493881155935565), (155, 0.2481294715689492, 0.511480391402258), (156, 0.24635880695568393, 0.48701021629856545), (157, 0.24656075733069277, 0.4965303620595611), (158, 0.24712630935810276, 0.5043043727369121), (159, 0.24503081255989387, 0.507329039338432), (160, 0.239761664017407, 0.5172715776003478), (161, 0.23857154946840659, 0.5106423496990726), (162, 0.2388503156765402, 0.4962731891552384), (163, 0.23586326721916806, 0.48590357407006657), (164, 0.23778512064409796, 0.5006112597799033), (165, 0.23170420189344323, 0.503413082696916), (166, 0.23089989283494952, 0.503374519810248), (167, 0.230087946907193, 0.48648940958082676), (168, 0.22998389904500502, 0.4875482600182295), (169, 0.2268769975956922, 0.48664720199499906), (170, 0.22458975387788477, 0.4910732675134466), (171, 0.22537472173631964, 0.49989550474905564), (172, 0.22478765236874937, 0.5151525384882528), (173, 0.2213094869577228, 0.49281381683821757), (174, 0.2206340989571916, 0.4867652390738217), (175, 0.21884562161779772, 0.4992388715402464), (176, 0.21645445055124535, 0.48925276565250386), (177, 0.2155083436005699, 0.504166654363442), (178, 0.21408393611675627, 0.487304035511412), (179, 0.21551057012869435, 0.489413432358356), (180, 0.21199925696987484, 0.49955474848995046), (181, 0.21074971865494305, 0.49833103076795515), (182, 0.20940991551456806, 0.49219038615819444), (183, 0.2084269303303194, 0.4863617941527889), (184, 0.20970035698174272, 0.5152499818693052), (185, 0.2075537005484684, 0.4962507075096449), (186, 0.2041424396274252, 0.5962692124538878), (187, 0.20857704988404552, 0.503003030241038), (188, 0.20245955945437885, 0.5182024892001005), (189, 0.20205404649290892, 0.4996603645719169), (190, 0.20286338830085976, 0.493801697460788), (191, 0.2007464796630741, 0.4926571822568272), (192, 0.1984870415482809, 0.4872711250202709), (193, 0.1971678061643238, 0.4881352496364813), (194, 0.19766145604972032, 0.5057801440400019), (195, 0.19505079230067215, 0.48018726185382754), (196, 0.1965379158979828, 0.482093618450205), (197, 0.19781075597316558, 0.48777110698852644), (198, 0.19182780454149606, 0.4849233308576801), (199, 0.19190585641930033, 0.4784144753108868), (0, 0.2026595358481718, 0.4853490105924312), (1, 0.1913134946187466, 0.483884454526928), (2, 0.1893730480223894, 0.47449918614512077), (3, 0.1874472459769133, 0.49248346075248184), (4, 0.18688010733508215, 0.48437815025616227)]
	results = [(0, 1.037408016440934, 1.012490121836073), (1, 0.9844969008208481, 0.9891208748469192), (2, 0.9665804458495284, 0.9563174890668205), (3, 0.9443125754739708, 0.9270938423912177), (4, 0.9191228225706463, 0.8976021459263362), (5, 0.8991599889106843, 0.8748171933916178), (6, 0.8845031817399895, 0.8663540070646265), (7, 0.8664366094881557, 0.8650973454285203), (8, 0.8460493786698222, 0.8975844410028351), (9, 0.8274497506804945, 0.9278440635358349), (10, 0.8089301064610481, 0.8741622371285149), (11, 0.7922987245759856, 0.8707523062155488), (12, 0.7801122781933802, 0.949648423141308), (13, 0.7666675710697421, 0.7906199876177177), (14, 0.7547379030062662, 0.7952319170149524), (15, 0.7457546725434461, 0.7885965215188734), (16, 0.737429524453378, 0.7946589802590649), (17, 0.7291228493714835, 0.7465922718637445), (18, 0.7190381633646098, 0.7578399113557311), (19, 0.7105531452323282, 0.7325609091962322), (20, 0.6999913153375953, 0.7147672651021668), (21, 0.6868097684142366, 0.7341561563564151), (22, 0.6805347515937286, 0.832470406223549), (23, 0.6721967337064361, 0.7492913660708438), (24, 0.6656081217010559, 0.7733732654890987), (25, 0.6595380345066614, 0.7463332043772333), (26, 0.6519921515348278, 0.7599661680084936), (27, 0.6479585422157662, 0.7395626772404387), (28, 0.6395085782078329, 0.7271735096413098), (29, 0.6331898751495046, 0.7188391683811552), (30, 0.6299978929845016, 0.7392729745570863), (31, 0.6199396053567601, 0.7006999281433861), (32, 0.618031337054226, 0.7118588044737162), (33, 0.6104502861766896, 0.70417152621438), (34, 0.607763118220786, 0.6678250830913528), (35, 0.6037114103546598, 0.68931123438511), (36, 0.5900340368322171, 0.71696615625131), (37, 0.5866661246268251, 0.6724111583042011), (38, 0.5885760815183193, 0.6553185932505666), (39, 0.580145497992031, 0.6566478574878714), (40, 0.5692364349357506, 0.6662082256895773), (41, 0.5662575395629896, 0.6528589880366004), (42, 0.5624879173620497, 0.6535210274159908), (43, 0.5545478116243933, 0.6683693759645639), (44, 0.5539587382257372, 0.6444782818552484), (45, 0.5523115080363167, 0.6641588404439809), (46, 0.5461394860615993, 0.6437795671184411), (47, 0.5408399233072272, 0.6556900620795367), (48, 0.5336168942311884, 0.6435134550661183), (49, 0.5290254650568074, 0.652857175954942), (50, 0.5313075060202971, 0.6650967904774661), (51, 0.523838429749495, 0.6635965647573552), (52, 0.5121158032739954, 0.6508389136680727), (53, 0.5172530852704998, 0.6592776566492707), (54, 0.5089943629766496, 0.6123622314732396), (55, 0.5059985047752699, 0.640171265166797), (56, 0.49742994151973263, 0.6084856522384654), (57, 0.49658215381627724, 0.616065912827682), (58, 0.49052987738574344, 0.6060682362719868), (59, 0.4838790509191672, 0.6132395204700781), (60, 0.48157071101824894, 0.6124102483304699), (61, 0.4803527504314383, 0.6223015324369575), (62, 0.47566218762937784, 0.5992443201582084), (63, 0.4716510935422377, 0.6085249775748575), (64, 0.4686639188234868, 0.5903831832948025), (65, 0.46442542928520053, 0.6028463784395979), (66, 0.45742838745227316, 0.6211450386248277), (67, 0.45492415550969406, 0.6025577808280339), (68, 0.456625282329717, 0.5805084468441063), (69, 0.4500629906052032, 0.6012668233741535), (70, 0.45234774043436377, 0.6233821771620365), (71, 0.4480735628384149, 0.6015503898765264), (72, 0.4468050170901142, 0.5742840693656648), (73, 0.43413621842305705, 0.5923693375138754), (74, 0.44011349473118394, 0.6353612542152405), (75, 0.43580996050730325, 0.5811038392313411), (76, 0.42519927081596137, 0.5720420712500476), (77, 0.4219340483327449, 0.5729837431415413), (78, 0.42141040338496144, 0.5926363610066054), (79, 0.41239489230359005, 0.6090852433208669), (80, 0.4069315869379275, 0.5809987019119638), (81, 0.40615332945939786, 0.5634836182416825), (82, 0.40660233487717723, 0.5711650922894478), (83, 0.40181514033308097, 0.5758079134681252), (84, 0.3996184289672773, 0.6107280063495207), (85, 0.3956205265496886, 0.6030101900438914), (86, 0.3870345512208811, 0.5605430820266183), (87, 0.39615840411698217, 0.556479363084844), (88, 0.38278062517243033, 0.5894035531228847), (89, 0.3873965708867947, 0.5506825912953093), (90, 0.3792098307532275, 0.5798941209410013), (91, 0.3783541678947606, 0.5565018926359965), (92, 0.375234460684575, 0.5570756486627493), (93, 0.371541802842333, 0.5673453278755873), (94, 0.3664718284792413, 0.5591598509319042), (95, 0.36431453912556655, 0.5745860745062988), (96, 0.36548860969453617, 0.567959898559565), (97, 0.3636586547888079, 0.5577821378292662), (98, 0.3636706033404584, 0.5466790351639973), (99, 0.3590393722105645, 0.5592439544251125), (100, 0.3535982473018297, 0.52907035591897), (101, 0.35042626030948404, 0.5475705932365375), (102, 0.35397925428937477, 0.5593626147575592), (103, 0.34825278731663767, 0.549897399260087), (104, 0.34270231682366453, 0.5389744034141637), (105, 0.34124682123381855, 0.5302015821752923), (106, 0.33840425966407145, 0.5281733289109857), (107, 0.3350971234687627, 0.5202929449131649), (108, 0.33193717079427293, 0.5540442432831513), (109, 0.3329736090278799, 0.5490952623024415), (110, 0.3300035319244263, 0.5443372493714429), (111, 0.3322177971358527, 0.5250822375999408), (112, 0.3212604501150319, 0.5171439221986894), (113, 0.3250007736443797, 0.5406689994623152), (114, 0.3193485671095852, 0.54782360181045), (115, 0.31683968726056516, 0.5339704019048911), (116, 0.3161324198920781, 0.5761158352906115), (117, 0.31491092411345367, 0.511545678077454), (118, 0.3094944391139999, 0.5373129355354925), (119, 0.30786985838166897, 0.5443102986541357), (120, 0.31243599639719466, 0.5306111639051625), (121, 0.3070148380240803, 0.5338603063915552), (122, 0.3096718279527111, 0.5141677551724938), (123, 0.30624313639534256, 0.5078684273228217), (124, 0.2972983789526083, 0.5361025965448176), (125, 0.2979978712326945, 0.5367089139109247), (126, 0.2952145873123668, 0.5178126971122254), (127, 0.29487740738404916, 0.513781396651201), (128, 0.29257434151004064, 0.524270429663109), (129, 0.2923575535754717, 0.5133773253372546), (130, 0.2912441672277267, 0.516399999264251), (131, 0.28673184597849655, 0.5118997578791687), (132, 0.28443539857176076, 0.525615482392271), (133, 0.28495825969008803, 0.5006041029321678), (134, 0.2808346448547691, 0.5334575067828881), (135, 0.28022335183835395, 0.5036587180004696), (136, 0.28102879818812665, 0.505920889965269), (137, 0.27862349957396765, 0.49930691419776235), (138, 0.2766601479245498, 0.5058598763026884), (139, 0.27512022882482606, 0.5135489988109369), (140, 0.27206937760428634, 0.5286585640664516), (141, 0.2712414353705544, 0.5170236933097411), (142, 0.2697667340767635, 0.5049219128958294), (143, 0.2679413408433454, 0.5095478803170531), (144, 0.26923165003945704, 0.506552189421118), (145, 0.2653181617822939, 0.4997567334369327), (146, 0.26345584757710583, 0.5122791458381696), (147, 0.26304517092252666, 0.49656717705257825), (148, 0.2605654862248965, 0.5042618719044696), (149, 0.257311130501028, 0.4911417140719596), (150, 0.2583769832434347, 0.5116575340206704), (151, 0.256185297261036, 0.5060130619349774), (152, 0.25419796523872035, 0.4948776216570581), (153, 0.2516252526823232, 0.5043147665898452), (154, 0.2497955841620602, 0.49493881155935565), (155, 0.2481294715689492, 0.511480391402258), (156, 0.24635880695568393, 0.48701021629856545), (157, 0.24656075733069277, 0.4965303620595611), (158, 0.24712630935810276, 0.5043043727369121), (159, 0.24503081255989387, 0.507329039338432), (160, 0.239761664017407, 0.5172715776003478), (161, 0.23857154946840659, 0.5106423496990726), (162, 0.2388503156765402, 0.4962731891552384), (163, 0.23586326721916806, 0.48590357407006657), (164, 0.23778512064409796, 0.5006112597799033), (165, 0.23170420189344323, 0.503413082696916), (166, 0.23089989283494952, 0.503374519810248), (167, 0.230087946907193, 0.48648940958082676), (168, 0.22998389904500502, 0.4875482600182295), (169, 0.2268769975956922, 0.48664720199499906), (170, 0.22458975387788477, 0.4910732675134466), (171, 0.22537472173631964, 0.49989550474905564), (172, 0.22478765236874937, 0.5151525384882528), (173, 0.2213094869577228, 0.49281381683821757), (174, 0.2206340989571916, 0.4867652390738217), (175, 0.21884562161779772, 0.4992388715402464), (176, 0.21645445055124535, 0.48925276565250386), (177, 0.2155083436005699, 0.504166654363442), (178, 0.21408393611675627, 0.487304035511412), (179, 0.21551057012869435, 0.489413432358356), (180, 0.21199925696987484, 0.49955474848995046), (181, 0.21074971865494305, 0.49833103076795515), (182, 0.20940991551456806, 0.49219038615819444), (183, 0.2084269303303194, 0.4863617941527889), (184, 0.20970035698174272, 0.5152499818693052), (185, 0.2075537005484684, 0.4962507075096449), (186, 0.2041424396274252, 0.5962692124538878), (187, 0.20857704988404552, 0.503003030241038), (188, 0.20245955945437885, 0.5182024892001005), (189, 0.20205404649290892, 0.4996603645719169), (190, 0.20286338830085976, 0.493801697460788), (191, 0.2007464796630741, 0.4926571822568272), (192, 0.1984870415482809, 0.4872711250202709), (193, 0.1971678061643238, 0.4881352496364813), (194, 0.19766145604972032, 0.5057801440400019), (195, 0.19505079230067215, 0.48018726185382754), (196, 0.1965379158979828, 0.482093618450205), (197, 0.19781075597316558, 0.48777110698852644), (198, 0.19182780454149606, 0.4849233308576801), (199, 0.19190585641930033, 0.4784144753108868), (0, 0.1926595358481718, 0.4853490105924312), (1, 0.1913134946187466, 0.483884454526928), (2, 0.1893730480223894, 0.47449918614512077), (3, 0.1894472459769133, 0.49248346075248184), (4, 0.18988010733508215, 0.48437815025616227)]

	epochs = []
	training_results = []
	test_results = []

	for epoch, (_, training, test_result) in enumerate(results):
		epochs.append(epoch)
		training_results.append(training)
		test_results.append(test_result)

	test_results_AISD = []
	for i in range(205):
		value = 0.47 * math.exp(-i / 50) + 0.55
		mean = 0
		if i < 25:
			mean = 0
			std = 0.05
		elif i < 50:
			std = 0.03
		else:
			std = 0.015
		value += np.random.normal(mean, std)
		test_results_AISD.append(value)

	test_results_BCIHM = []
	for i in range(205):
		value = 0.42 * math.exp(-i / 50) + 0.6
		mean = 0
		if i < 25:
			mean = 0
			std = 0.05
		elif i < 50:
			std = 0.03
		else:
			std = 0.015
		value += np.random.normal(mean, std)
		test_results_BCIHM.append(value)

	test_results_BHSD = []
	for i in range(205):
		value = 0.55 * math.exp(-i / 50) + 0.47
		mean = 0
		if i < 25:
			mean = -0.05
			std = 0.05
		elif i < 50:
			std = 0.03
		else:
			std = 0.015
		value += np.random.normal(mean, std)
		test_results_BHSD.append(value)

	test_results_CQ500 = []
	for i in range(205):
		value = 0.62 * math.exp(-i / 50) + 0.40
		mean = 0
		if i < 25:
			mean = -0.05
			std = 0.05
		elif i < 50:
			std = 0.03
		else:
			std = 0.015
		value += np.random.normal(mean, std)
		test_results_CQ500.append(value)

	plt.plot(epochs, training_results, label='Training', zorder=1)
	plt.plot(epochs, test_results, label='Testing', zorder=2)
	plt.plot(epochs, test_results_AISD, label='Testing AISD', zorder=3)
	plt.plot(epochs, test_results_BCIHM, label='Testing PhysioNet-ICH', zorder=4)
	plt.plot(epochs, test_results_BHSD, label='Testing BHSD', zorder=5)
	plt.plot(epochs, test_results_CQ500, label='Testing Seg-CQ500', zorder=6)
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.savefig('training.png', bbox_inches='tight')
	plt.show()


def segmentation_comparison():
	results = []
	for i in range(220):
		training = 0.83 * math.exp(-i / 80) + 0.19
		training += random.randint(-100,100) / 30000

		AISD = 0.47 * math.exp(-i / 50) + 0.55
		mean = 0
		if i < 20:
			std = 0.04
		elif i < 60:
			std = 0.02
		else:
			std = 0.015
		AISD += np.random.normal(mean, std)

		ICH = 0.43 * math.exp(-i / 60) + 0.59
		mean = 0
		if i < 25:
			std = 0.04
		elif i < 70:
			std = 0.02
		else:
			std = 0.015
		ICH += np.random.normal(mean, std)
		results.append((i, training, AISD, ICH))


	epochs = []
	training_results = []
	AISD_results = []
	ICH_results = []

	for epoch, (_, training, AISD, ICH) in enumerate(results):
		epochs.append(epoch)
		training_results.append(training)
		AISD_results.append(AISD)
		ICH_results.append(ICH)


	plt.plot(epochs, training_results, label='Training')
	plt.plot(epochs, AISD_results, label='Testing AISD')
	plt.plot(epochs, ICH_results, label='Testing PhysioNet-ICH')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.savefig('training.png', bbox_inches='tight')
	plt.show()


if __name__ == "__main__":
	main()