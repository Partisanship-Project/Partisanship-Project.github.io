var divs=[];
var cardrows=[];
//generates a list of members - ["John Kennedy", "John McCain"]
function getMembers(){
    window.data=[];
    window.words=[];
    var members=[];
    d3.csv("../assets/replication/Data/metadata.csv", function(data) {
        members.push(data.name);
        window.data.push(data);
    });
    window.words=[];
    d3.csv("../assets/replication/Results/PartisanWordsPerMember.csv", function(data) {
        window.words.push(data);
    });
    return members
}

function refreshMemberPage(){
    //identify who they're looking for
    var newdata=[];
    var member_input=$(".member_select").val();
    if (member_input=='' || member_input=='Select a Member'){
        member_input=$(".member_select").val("Select a Member");
    }  
    //clean out existing page
    $('#main_container').text('');
    //insert searched for member
    fetchMemberData();
    //find and insert competitors
    fetchCompetition(); 
    //find and insert related members
    fetchRelatedMemberData();
}

function fetchMemberData(){
    var newdata=[];
    var member_input=$(".member_select").val();
    if (member_input=='' || member_input=='Select a Member'){
        member_input=$(".member_select").val("Select a Member");
    }else{
        console.log('got here1')
        fetchRaceByName(member_input);
        //if (location.hostname === "localhost" || location.hostname === "127.0.0.1"|| location.hostname === ""){
        //    $('#official_name').text('successful name change');
        //    $('#affiliation').text('succesful party change');
        //}else{
        //    console.log('got here1')
        //    
        //    fetchRaceByName(member_input);
        //}
        //NEED CONDITION IF THEY ENTER AN INVALID NAME 
    }
    $(".member_select").val('')
}

function fetchRaceByName(name){
    var title=''; //[Chamber] [Full Name]
    var affiliation = ''; //[party] from [State]
    var party ='';
    var state= '';
    var district='';
    var divs=[];
    for (var i = 0; i < window.data.length; i++){
    //d3.csv("../assets/replication/Data/metadata.csv", function(data) {
        //console.log(data)
        //console.log(data.name)
        if (window.data[i].name==name){
            state=window.data[i].state;
            district=window.data[i].district;
            console.log('got here 2')
            console.log('state'+state);
            console.log('district '+district);
            console.log(data);
            //fetchRaceData(state, district, function(data) {
            //    showRaceData(data);
            //})
            fetchRaceData(state,district);
            //console.log('got here 5');
            //console.log(JSON.stringify(divs))
            //cardrows=showRaceData(divs);
        }
    }
    //});
    
}

function showRaceData(divs){
    console.log('got here 4')
    //Hide the carousel - NOT NECESSARY if we move to races.html
    $('#slider1-1').hide()
    
    //now that we have all the cards
    //clear the existing container
    $("#member_container").empty()
    $("#project_summary").hide()
    
    //var cardrows=[];
    //var card=$("<div class='col-"+ratio.toString()+" col-sm-"+ratio.toString()+"'>")
    //card.append(div)
    for (r=0; r<=Math.floor(divs.length/3); r++ ){
        var div =[];
        if (divs.length>3){
            div=divs.splice(0,3);           //get three per row
        }else{
            div=divs.splice(0,divs.length); //otherwise get all of them
        }
        var cards=[];
        
        var bigrow=$("<div class=row></div>");
        for (var k=0; k<=div.length; k++){
            console.log(JSON.stringify(div[k]))
            if (div.length<3){
                var card=$("<div class='col-6 col-sm-6'>");
            }else{
                var card=$("<div class='col-4 col-sm-4'>");
            }
            card.append(div[k]);
            bigrow.push(card);
        }
        cardrows.push(bigrow)
    console.log('got here final')
    console.log(cardrows)
    return cardrows
    //$("#member_container").append(cardrows);        
    }   
}

function fetchRaceData(state,district){
    console.log('got here 3')
    console.log('state'+state);
    console.log('district '+district);
    //idea here is person enters a zipcode and we lookup their district.
    var title=''; //[Chamber] [Full Name]
    var affiliation = ''; //[party] from [State]
    var party ='';
    cards=[];
    divs=[];
    for (var i = 0; i < window.data.length; i++){
    //d3.csv("../assets/replication/Data/metadata.csv", function(data) {
        var count=0;
        //console.log(data.state+' '+state+' '+data.district+' '+district)
        //console.log('got here 2')
        if (window.data[i].state==state & window.data[i].district==district){
            count++;
            console.log('getting data for ' + window.data[i].name )
            var headshot_district_slug='';
            if (window.data[i].district=='Senate'){
                title='Senator ';
                headshot_district_slug='Sen';
            } else {
                title="Representative ";
                if (window.data[i].district==0){
                    headshot_district_slug='';
                }else if(window.data[i].district<10 && window.data[i].district>0){
                    headshot_district_slug=0+window.data[i].district;
                }else{
                    headshot_district_slug=window.data[i].district;
                }
            }
            
            if (window.data[i].party=='D'){
                party='<span style="color: #0000a0">Democratic</span> Candidate';
            }else if (window.data[i].party=='R'){
                party='<span style="color: #ff0000">Republican</span> Candidate';
            }else{
                party='Candidate for ';
            }
            
            var headshot_slug=window.data[i].state+headshot_district_slug+'_'+window.data[i].party;
            

            var headshot_url='../assets/images/headshots/'+headshot_slug+'.jpg';
            
            //headshot_url='../assets/images/headshots/sinclair.jpg';
            
            //create title
            var h=$("<h5 style='text-align: center'></h5>");
            var n=$("<div id='official_name"+i.toString()+"'>").text(window.data[i].name); //.text(title.concat(window.data[i].name));
            h.append(n);
            
            //create and append first row of profile picture and member info
            var row1=$("<div class=row></div>")
            var col1=$("<div class='col-6 col-sm-6'>")
            var img1=$('<img src='+headshot_url+' style="width: 100%;"  id="picture_'+count.toString()+'" title="">');
            col1.append(img1)
            var col2=$("<div class='col-6 col-sm-6'>")
            var af=$('<div id="affiliation_'+count.toString()+'"></div>').html(party);
            col2.append(af)
            row1.append(col1,col2)
            
            //create and append second row of history of partisanship
            var row2=$("<div class=row style='padding-top: 15px'></div>")
            var col3=$("<div class='col' style='text-align: center'>")
            if (window.data[i].twitter!=''){
                var history_url='/assets/replication/Images/'+window.data[i].twitter+'.jpg';
                //history_url='/assets/replication/Images/sinclair.jpg';
                var photo=$('<img src='+history_url+' style="width: 75%;" data-toggle="tooltip" id="history_'+count.toString()+'" title="Estimated partisanship over time">');
                //var words=getWords(data.twitter);
            }else{
                var photo=$("<h5>No Twitter Account Found</h5>");
                //var history_url='/assets/replication/Images/no_data.jpg';       //need to create this image
                //history_url='/assets/replication/Images/sinclair.jpg';
            }
            
            col3.append(photo)
            row2.append(col3)
            
            //create and append table of top partisan words
            var row3=$("<div class=row></div>")
            
            //FORMAT TOPWORDS TABLE
            if (window.data[i].twitter==''){
                var wordtable=$("<h5>No Twitter Account Found</h5>");
            }else{
                var wordtable=getWordTable(window.data[i].twitter)
            }
            row3.append('top words table!')
            var div=$("<div></div>");
            div.append(h,row1,row2,wordtable)
            divs.push(div)
            
        }
    }
    console.log('number of cards')
    console.log(divs.length)
    //Hide the carousel
    $('#slider1-1').hide()
    
    //now that we have all the cards
    //clear the existing container
    $("#member_container").empty()
    
    //Center heading for results
    var main= $("<h2 style='text-align:center'>Candidates for "+ state+ " "+district +" </h5>");
    var cardrows=[];
    var cards=[];
    console.log('checking multirow results')
    for (r=0; r<=Math.floor(divs.length/3); r++ ){
        
        console.log('r='+r)
        //var div=$("<div class=row id=testdiv></div>");
        if (divs.length>3){
            row=divs.splice(0,3);           //get three per row
        }else{
            row=divs.splice(0,divs.length); //otherwise get all of them
        }
        var bigrow=$("<div class=row ></div>");
        for (var k=0; k<row.length; k++){
            console.log('k='+k)
            if (row.length>=3){
                var card=$("<div class='col-lg-4 col-md-6 col-sm-12' style='padding-bottom: 20px'>");
            }else{
                var card=$("<div class='col-lg-6 col-md-6 col-sm-12' style='padding-bottom: 20px'>");
            }
            card.append(row[k]);
            cards.push(card);
            console.log(cards.length)
        
        }
    bigrow.append(cards);
    cardrows.push(bigrow);
    }
    console.log(JSON.stringify(cardrows))
    $("#member_container").append(main,cardrows);
              
                
    //});
    //};
    //console.log(JSON.stringify(divs))
    //return divs
    //for (d in div)
    //return output
}

function getWordTable(twitter){
    //function takes an array of words and returns an html table
    console.log('getting words for ' + twitter)
    var  phrases=[];
    for (var i = 0; i < window.words.length; i++){
        //console.log(window.words[i].TwitterID+" "+twitter)
        if (window.words[i].TwitterID==twitter.toLowerCase()){
            var raw=window.words[i].words;
            raw=raw.replace(new RegExp('mnton_', 'g'), '@');
            raw=raw.replace(new RegExp('hshtg_', 'g'), '#');
            //raw=raw.split('mnton_').join('@');
            phrases=raw.split(',')
            console.log(twitter+ ' '+ phrases)
            
        }
    }
    var numwordsperrow=3;
    var numcols=Math.floor(phrases.length/numwordsperrow);
    var rows=[];
    for (r=0; r<numcols; r++ ){
        console.log(numcols)
        var Row=$("<tr></tr>");
        cells=[]
        if (phrases.length>numwordsperrow){
            var tablerow=phrases.splice(0,numwordsperrow);           //get three per row
        }else{
            var tablerow=phrases.splice(0,phrases.length); //otherwise get all of them
        }
        for (k=0; k<tablerow.length; k++){
            var Cell=$("<td style='padding:5px'></td>");
            Cell.text(tablerow[k])
            cells.push(Cell)
            //var cell=words[i]
        }
        Row.append(cells)
        rows.push(Row)
        
    }
    var h=$("<h5 style='text-align: center'>Most Partisan Phrases this Year</h5>");
    var table=$("<table class=words style='border: 1px solid black; margin: 0 auto' data-toggle='tooltip' title='These are the words used by this candidate that our algorithm predicts are also associated with their party'></table>");
    table.append(rows)
    var div=$("<div class=row style='padding-top: 15px'></div>");
    var col=$("<div class=col style='text-align: center'></div>");
    col.append(h,table)
    div.append(col)
    return div
}
function unitTest(){
    
}