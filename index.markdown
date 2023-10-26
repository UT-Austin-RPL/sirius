---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Robot Learning on the Job</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Merriweather","PT Serif",Georgia,"Times New Roman",serif;
    font-weight: 300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
    color: #333332;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }
  h3 {
    font-weight:250;
    font-size: 25px;
  }
  
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}

hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->

<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">
</head>

<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center>
<h1><strong>
<!-- Ditto <img width="50" style='display:inline;' src="./src/ditto.png"/> <br> -->
Robot Learning on the Job: Human-in-the-Loop Manipulation and Learning During Deployment
</strong></h1></center>
<center><h2>
    &nbsp;&nbsp;&nbsp;<a href="https://huihanl.github.io/">Huihan Liu</a>&nbsp;&nbsp;
    <a href="http://snasiriany.me/">Soroush Nasiriany</a>&nbsp;&nbsp; 
    Lance Zhang&nbsp;&nbsp;
    Zhiyao Bao&nbsp;&nbsp;
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp; 		
    </h2></center>
    <center><h2>
        in submission to ICRA 2023 &nbsp;&nbsp;&nbsp; 		
    </h2></center>
	<center><h2><a href="https://arxiv.org/abs/2202.08227">Paper</a> | <a href="https://github.com/UT-Austin-RPL/Ditto">Code</a> </h2></center>


<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
 With the rapid growth of computing powers and recent advances in deep learning, we have witnessed impressive demonstrations of novel robot capabilities in research settings. Nonetheless, these learning systems exhibit brittle generalization and require excessive training data for practical tasks. To harness the capabilities of state-of-the-art robot learning models, while embracing their imperfections, we develop a principled framework for humans and robots to collaborate through a division of work. In this framework, partially autonomous robots are tasked with handling a major portion of decision-making where they work reliably; meanwhile, human operators monitor the process and intervene in challenging situations. Such a human-robot team ensures safe deployments in complex tasks. Further, we introduce a new learning algorithm to improve the policy's performance on the data collected from the task executions. The core idea is re-weighing training samples with approximated human trust and optimizing the policies with weighted behavioral cloning. We evaluate our framework in simulation and on real hardware, showing that our method consistently outperforms baselines over a collection of contact-rich manipulation tasks, achieving 8% boost in simulation and 27% on real hardware than the state-of-the-art methods, with 3 times faster convergence and 15% memory size.
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">Learning on the Job Overview</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay loop width="100%">
        <source src="./video/1_overview.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

<hr>

<h1 align="center">Continuous Deployment and Update Cycle</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay loop width="100%">
        <source src="./video/2_cicd.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

<hr>

<h1 align="center">Method: Intervention-based Reweighting Scheme</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay loop width="100%" frameborder="5">
        <source src="./video/3_model.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

<hr>

<h1 align="center">Experiment Results</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">Our system ensures safe and reliable execution through human-robot teaming. We evaluated the autonomous policy performance of our human-in-the-loop framework on 4 tasks. As the autonomous policy improves over long-term deployment, the amount of human workload decreases.
</p>
</td>
</tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay width="100%">
      <source src="./video/5_tasks.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">We conduct 3 rounds of robot deployments and policy updates. Here we present Round 1 and Round 3 results of Ours and baseline IWR. We show how for Ours policy performance improve over rounds, and how Ours outperforms IWR baseline. </p>
</td>
</tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay loop width="100%">
      <source src="./video/6_timeline.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>

<h2 align="center">Gear Insertion (Real)</h2>

<h3 align="center">Ours, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g1.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>


<h3 align="center">IWR, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g2.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">Ours, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g3.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">IWR, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g4.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>

<h2 align="center">Coffee Pod packing (Real)</h2>

<h3 align="center">Ours, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g5.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">IWR, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g6.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">Ours, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g7.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">IWR, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted   width="100%">
      <source src="./video/g8.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>
  </tbody>
</table>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@inproceedings{jiang2022ditto,
   title={Ditto: Building Digital Twins of Articulated Objects from Interaction},
   author={Jiang, Zhenyu and Hsu, Cheng-Chun and Zhu, Yuke},
   booktitle={arXiv preprint arXiv:2202.08227},
   year={2022}
}
</code></pre>
</left></td></tr></table>

<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> We would like to thank Yifeng Zhu for help on real robot experiments. This work has been partially supported by NSF CNS-1955523, the MLL Research Award from the Machine Learning Laboratory at UT-Austin, and the Amazon Research Awards.
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->
