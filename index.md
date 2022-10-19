---
mailinglist: 
---

<head>
  <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
  <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.13/js/jquery.dataTables.min.js"></script>
  <script type="text/javascript" charset="utf8" src="leaderboard.js"></script>

</head>
<style>
  .table.dataTable  {
    font-family: Verdana, Geneva, Tahoma, sans-serif;
    font-size: 12px;
}
</style>


[paper]: https://doi.org/10.48550/arxiv.2210.04573
[repository]: https://github.com/the-deep/humset

HumSet is a novel and rich multilingual dataset of humanitarian response documents annotated by experts in the humanitarian response community. HumSet is curated by humanitarian analysts and covers various disasters around the globe that occurred from 2018 to 2021 in 46 humanitarian response projects. The dataset consists of approximately 17K annotated documents in three languages of English, French, and Spanish, originally taken from publicly-available resources. For each document, analysts have identified informative snippets (entries) in respect to common humanitarian frameworks, and assigned one or many classes to each entry. See the our pre-print short paper for details.

**Paper:** [Humset - Dataset of Multilingual Information Extraction and Classification for Humanitarian Crisis Response][paper]
```
@misc{https://doi.org/10.48550/arxiv.2210.04573,
  doi = {10.48550/ARXIV.2210.04573},
  url = {https://arxiv.org/abs/2210.04573},
  author = {Fekih, Selim and Tamagnone, Nicolò and Minixhofer, Benjamin and Shrestha, Ranjan and Contla, Ximena and Oglethorpe, Ewan and Rekabsaz, Navid},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {HumSet: Dataset of Multilingual Information Extraction and Classification for Humanitarian Crisis Response},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
<br/><br/>
* [Dataset](#dataset)
  * [Additional data](#addittional-data)

* [Request access](#request-access)
* [Contact](#contact)
* [Terms and conditions](#terms-and-conditions)
<br/><br/>

### Dataset 

Main dataset is shared in CSV format (<em>**humset_data.csv**</em>), where each row is considered as an <em>entry</em> with the following features: 

<div class="alert bg-success text-dark" cellspacing="0" style="width:100%">
  <table id="leaderboard_head_dctr" class="table table-bordered" cellspacing="0">
    <thead>
      <tr><th>entry_id</th><th>lead_id</th><th>project_id</th><th>sectors</th><th>pillars_1d</th><th>pillars_2d</th><th>subpillars_1d</th><th>subpillars_2d<th>lang</th><th>n_tokens</th><th>project_title</th><th>created_at</th><th>document</th><th>excerpt</th></tr>
    </thead>
  </table>
</div>

- **entry_id**: tpyeunique identification number for a given entry. (int64)
- **lead_id**: unique identification number for the document to which the corrisponding entry belongs. (int64)
- **sectors**, **pillars_1d**, **pillars_2d**, **subpillars_1d**, **subpillars_2d**: labels assigned to the corresponding entry. Since this is a multi-label dataset (each entry may have several annotations belonging to the same category), they are reported as arrays of strings. For a detailed description of these categories, see the [paper]. (list)
- **lang**: language. (str)
- **n_tokens**: number of tokens (tokenized using NLTK v3.7 library). (int64)
- **project_title**: the name of the project where the corresponding annotation was created. (str)
- **created_at**: date and time of creation of the annotation in stardard ISO 8601 format. (str)
- **document**: document URL source of the excerpt. (str)
- **excerpt**: excerpt text. (str)

**Note**: 
- **subpillars_1d** and **subpillars_2d** respective tags are reported, as strings, with the format {PILLAR}->{SUBPILLARS}, in order to underline the hierarchical structure of 1D and 2D categories. 
<br/><br/>
#### Addittional data

In addition to the main dataset, documents (<em>leads</em>) full texts are also reported (<em>**documents.tar.gz**</em>). Each text source is represented JSON-formatted file ({**lead_id**}.json) with the following structure: 
```
[
  [
    paragraph 1 - page 1,
    paragraph 2 - page 1,
    ...
    paragraph N - page 1
  ],
  [
    paragraph 1 - page 2,
    paragraph 2 - page 2,
    ...
    paragraph N - page 2
  ],
  [
    ...
  ],
  ...
]
```
Each document is a list of lists of strings, where each element is the text of a page, divided into the corresponding paragraphs. This format was used since, as indicated in the [paper], over 70% of the sources are in PDF format, thus choosing to keep the original textual subdivision. In the case of HTML web pages, the text is reported as if it belongs to a single page document.
<br/><br/>
Additionally, <em>train/validation/test</em> splitted dataset is shared. The [repository] contains the code with which it is possible to process the total dataset, but the latter contains some random components which would therefore result in a slightly different result.


* Pyserini guideline for creating BM25 baselines: <a href="https://github.com/castorini/pyserini/blob/master/docs/experiments-tripclick-doc.md" target="_blank">link</a>
* A new set of training triples (`triples.train.tsv`) provided by Hofstätter et al.: <a href="https://github.com/sebastian-hofstaetter/tripclick" target="_blank">github</a>, <a href="https://huggingface.co/datasets/sebastian-hofstaetter/tripclick-training" target="_blank">training triples</a>

## Request access
To gain access to HumSet, please fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSesb1-GChU4IsUadhzyn8bJPn6usyaiICoqhqEivtkJF_zBEg/viewform)

## Contact
For any technical question please contact [Selim Fekih](mailto:selim@datafriendlyspace.org), [Nicolò Tamagnone]((mailto:nico@datafriendlyspace.org)).

<!---
<br>
<div class="row">
    <div class="col-md-4 text-center">
        <a target="_blank" href="https://www.jku.at/en/institute-of-computational-perception/about-us/people/navid-rekab-saz/"><img src="images/navid.png" width="150" height="150"><br><strong>Navid Rekab-saz</strong><br>Johannes Kepler University Linz</a>
    </div>
    <div class="col-md-4 text-center">
        <a target="_blank" href="https://www.jku.at/en/institute-of-computational-perception/about-us/people/oleg-lesota/"><img src="images/oleg.webp" width="150" height="150"><br><strong>Oleg Lesota</strong><br>Johannes Kepler University Linz</a>
    </div>
    <div class="col-md-4 text-center">
        <a target="_blank" href="https://www.jku.at/en/institute-of-computational-perception/about-us/people/markus-schedl"><img src="images/markus.jpg" width="87" height="150"><br><strong>Markus Schedl</strong><br>Johannes Kepler University Linz</a>
    </div>
</div>
<br>
<div class="row">
    <div class="col-md-6 text-center">
        <a target="_blank" href="mailto:jon.brassey@tripdatabase.com?subject=[TripClick]"><img src="images/jon.webp" width="150" height="150"><br><strong>Jon Brassey</strong><br>Trip Database</a>
    </div>
    <div class="col-md-6 text-center">
        <a target="_blank" href="https://brown.edu/Research/AI/people/carsten.html"><img src="images/carsten.png" width="150" height="150"><br><strong>Carsten Eickhoff</strong><br>Brown University</a>
    </div>
</div>
--->

### Terms and conditions
The provided datasets are intended for non-commercial research purposes to promote advancement in the field of natural language processing, information retrieval and related areas, and are made available free of charge without extending any license or other intellectual property rights. In particular:
* Any parts of the datasets cannot be publicly shared or hosted (with exception for aggregated findings and visualizations);
* The datasets can only be used for non-commercial research purposes;
* The statistical models or any further resources created based on the datasets cannot be shared publicly without the permission of the data owners. These include for instance the weights of deep learning models trained on the provided data.

Upon violation of any of these terms, my rights to use the dataset will end automatically. 
The datasets are provided “as is” without warranty. The side granting access to the datasets is not liable for any damages related to use of the dataset.

<br/><br/>
<img src="images/dfs-logo-full-color-rgb.svg" alt="DFS logo" width="90"/>

<script>
  $(function(){
    var otable_leaderboard_head_dctr = $("#leaderboard_head_dctr").dataTable({
        bAutoWidth: false, 
        bPaginate: false,
        sScrollX: "100%",
        bInfo : false,
        sDom: 'l<"toolbar">frtip',
        aoColumns: [
          { sWidth: '5%' },
          { sWidth: '35%' },
          { sWidth: '35%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' }
        ],      
        aaData:data_head_dctr
    });
    otable_leaderboard_head_dctr.fnSort( [ [5,'desc'] ] );
    var otable_leaderboard_head_raw = $("#leaderboard_head_raw").dataTable({
        bAutoWidth: false, 
        bPaginate: false,
        sScrollX: "100%",
        bInfo : false,
        sDom: 'l<"toolbar">frtip',
        aoColumns: [
          { sWidth: '5%' },
          { sWidth: '35%' },
          { sWidth: '35%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' }
        ],      
        aaData:data_head_raw
    });
    otable_leaderboard_head_raw.fnSort( [ [5,'desc'] ] );
    var otable_leaderboard_torso_raw = $("#leaderboard_torso_raw").dataTable({
        bAutoWidth: false, 
        bPaginate: false,
        sScrollX: "100%",
        bInfo : false,
        sDom: 'l<"toolbar">frtip',
        aoColumns: [
          { sWidth: '5%' },
          { sWidth: '35%' },
          { sWidth: '35%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' }
        ],      
        aaData:data_torso_raw
    });
    otable_leaderboard_torso_raw.fnSort( [ [5,'desc'] ] );
    var otable_leaderboard_tail_raw = $("#leaderboard_tail_raw").dataTable({
        bAutoWidth: false, 
        bPaginate: false,
        sScrollX: "100%",
        bInfo : false,
        sDom: 'l<"toolbar">frtip',
        aoColumns: [
          { sWidth: '5%' },
          { sWidth: '35%' },
          { sWidth: '35%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' },
          { sWidth: '5%' }
        ],      
        aaData:data_tail_raw
    });
    otable_leaderboard_tail_raw.fnSort( [ [5,'desc'] ] );
  })  
  
</script>
