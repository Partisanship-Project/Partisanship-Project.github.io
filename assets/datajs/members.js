//generates a list of members - ["John Kennedy", "John McCain"]
function getMembers(){
    var members=[];
    d3.csv("../assets/data/metadata.csv", function(data) {
        members.push(data.Fullname)
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
            fetchMetaData(member_input);
            fetchOtherMembers(member_input);
        }
        //$('official_name').text(newdata[0].Fullname);
    }
    fetchOtherMembers();
}

function fetchMetaDataByName(name){
    var title=''; //[Chamber] [Full Name]
    var affiliation = ''; //[party] from [State]
    var party ='';
    var state= '';
    d3.csv("../assets/data/metadata.csv", function(data) {
        console.log(data)
        if (data.Fullname==name){
            console.log('getting data')
            if (data.District==0){
                title='Senator ';
            } else {
                title="Representative ";
            }
            if (data.Party=='D'){
                party='Democratic Candidate for ';
            }else if (data.Party=='R'){
                party='Republican Candidate for ';
            }else{
                party='Candidate for ';
            }
            //output.push(data)
            $('#official_name').text(title.concat(data.Fullname));
            $('#affiliation').text(party.concat(data.State));
        }
    });
    //return output
}

function fetchMetaDataByDistrict(state,district){
    //idea here is person enters a zipcode and we lookup their district.
    var title=''; //[Chamber] [Full Name]
    var affiliation = ''; //[party] from [State]
    var party ='';
    var state= '';
    d3.csv("../assets/data/metadata.csv", function(data) {
        console.log(data)
        if (data.Fullname==name){
            console.log('getting data')
            if (data.District==0){
                title='Senator ';
            } else {
                title="Representative ";
            }
            if (data.Party=='D'){
                party='Democratic Candidate for ';
            }else if (data.Party=='R'){
                party='Republican Candidate for ';
            }else{
                party='Candidate for ';
            }
            //output.push(data)
            $('#official_name').text(title.concat(data.Fullname));
            $('#affiliation').text(party.concat(data.State));
        }
    });
    //return output
}

function addProfile(){
    //function adds a new column to the row
}