//generates a list of members - ["John Kennedy", "John McCain"]
function getMembers(){
    var members=[];
    d3.csv("../assets/replication/Data/metadata.csv", function(data) {
        members.push(data.name)
    });
    return members
}

function refreshMemberPage(){
    //identify who they're looking for
    var newdata=[];
    var member_input=$("#member_select").val();
    if (member_input=='' || member_input=='Select a Member'){
        member_input=$("#member_select").val("Select a Member");
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
    var member_input=$("#member_select").val();
    if (member_input=='' || member_input=='Select a Member'){
        member_input=$("#member_select").val("Select a Member");
    }else{
        if (location.hostname === "localhost" || location.hostname === "127.0.0.1"|| location.hostname === ""){
            $('#official_name').text('successful name change');
            $('#affiliation').text('succesful party change');
        }else{
            console.log('got here1')
            
            fetchRaceByName(member_input);
        }
        //NEED CONDITION IF THEY ENTER AN INVALID NAME 
    }
}

function fetchRaceByName(name){
    var title=''; //[Chamber] [Full Name]
    var affiliation = ''; //[party] from [State]
    var party ='';
    var state= '';
    d3.csv("../assets/replication/Data/metadata.csv", function(data) {
        console.log(data)
        if (data.name==name){
            var state=data.state;
            var district=data.district
            fetchRaceData(state,district)
        }
    });
}

function fetchRaceData(state,district){
    //idea here is person enters a zipcode and we lookup their district.
    var title=''; //[Chamber] [Full Name]
    var affiliation = ''; //[party] from [State]
    var party ='';
    var state= '';
    var cards=[];
    d3.csv("../assets/replication/Data/metadata.csv", function(data) {
        console.log(data)
        var count=0;
        if (data.state==state & data.district==district){
            count++;
        }    
        //if there are more than 3 condidates, we're going to do multiple rows
        if (count>3){
            var ratio=Math.round(12/count);
        }else{
            var ratio=4;
        }
        var i=0;
        if (data.state==state & data.district==district){
            i++;
            console.log('getting data')
            if (data.district==0){
                title='Senator ';
            } else {
                title="Representative ";
            }
            if (data.party=='D'){
                party='Democratic Candidate for ';
            }else if (data.party=='R'){
                party='Republican Candidate for ';
            }else{
                party='Candidate for ';
            }
            var headshot_url='../assets/images/headshots/'+district+state+'.jpg';
            headshot_url='../assets/images/headshots/sinclair.jpg';
            if (data.twitter!=''){
                var history_url='/assets/replication/Images/'+data.twitter+'.jpg';
                history_url='/assets/replication/Images/sinclair.jpg';
                var words=getWords(data.twitter);
            }else{
                var history_url='/assets/replication/Images/no_data.jpg';       //need to create this image
                history_url='/assets/replication/Images/sinclair.jpg';
                var words = ['Not Available'];
            }
            //Create the card object we will attach a members' data to
            var card=$("<div class='col-"+ratio.toString()+" col-sm-"+ratio.toString()+"'>")
            //create title
            var h=$("<h2></h2>");
            var n=$("<div id='official_name"+i.toString()+"'>").text(title.concat(data.name));
            h.append(n);
            
            //create and append first row of profile picture and member info
            var row1=$("<div class=row></div>")
            var col1=$("<div class='col-6 col-sm-6'>")
            var img1=$('<img src='+headshot_url+' style="width: 75%;" id="picture_'+i.toString()+'" title="">');
            col1.append(img1)
            var col2=$("<div class='col-6 col-sm-6'>")
            var af=$('<div id="affiliation_'+i.toString()+'"></div>').text(party.concat(data.state))
            col2.append(af)
            row1.append(col1,col2)
            
            //create and append second row of history of partisanship
            var row2=$("<div class=row></div>")
            var col3=$("<div class='col'>")
            var img1=
            history_url
            var photo=$('<img src='+history_url+' style="width: 150%;" id="history_'+i.toString()+'" title="">');
            col3.append(photo)
            row2.append(col3)
            
            //create and append table of top partisan words
            var row3=$("<div class=row></div>")
            //FORMAT TOPWORDS TABLE
            var wordtable=getWordTable(data.twitter)
            row3.append('top words table!')
            div.append(h,row1,row2,row3)
            card.append(div)            
            cards.push(card);  //add this card to the list of cards.
        }
        //Hide the carousel
        $('#slider1-1').hide()
        
        //now that we have all the cards
        //clear the existing container
        $("#member_container").empty()
        
        var cardrows=[];
        for (r=0; r<=Math.floor(count/3); r++ ){
            if (cards.length>=3){
                var bigrow=$("<div class=row></div>")
                bigrow.append(cards.slice(0,3))
                cardrows.push(bigrow)
            }else if (cards.length!=0){
                var bigrow=$("<div class=row></div>")
                bigrow.append(cards)
                cardrows.push(bigrow)
                $("#member_container").append(cardrows)
            }else{
                $("#member_container").append(cardrows)
            }
            
        }
        
    });
    //for (d in div)
    //return output
}

function unitTest(){
    
}