
(function(){
  var my_node_id,curr_generation, replication, read_multiple_versions, stories, running_total_pay, num_stories_to_read, decision_index, participant_index;
  let read_stories = [];
  //var running_total_pay = 0;
  var loading_timeout = 500; // miliseconds next story is loaded (including the timeout smooths the loading process) 
  //var max_bonus = 1.5;
  let numKeyStrokes = 0;

  function hashCode(string){
    var hash = 0;
    for (var i = 0; i < string.length; i++) {
        var code = string.charCodeAt(i);
        hash = ((hash<<5)-hash)+code;
        hash = hash & hash; // Convert to 32bit integer
    }
    return hash;
  }

  function maxLength(el) {    
    if (!('maxLength' in el)) {
        var max = el.attributes.maxLength.value;
        el.onkeypress = function () {
            if (this.value.length >= max) return false;
        };
    }
  }

  function shuffle(array) {
    let currentIndex = array.length,  randomIndex;
    while (currentIndex != 0) {
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
    return array;
  }

  $(document).ready(function() {
    $('.disablecopypaste').bind('copy paste', function (e) {
      e.preventDefault();
    });

    maxLength(document.getElementById("reproduction"));

    // On every keydown add 1 to the number of keystrokes
    $('#reproduction').on('keydown', function() {
      numKeyStrokes++;
    });
      
    $("#submit-response").click(function() {
      $("#reproduction").prop('disabled', true);
      $("#submit-response").addClass('disabled');
      $("#submit-response").html('Submitting...');

      if (numKeyStrokes < $('#reproduction').val().length) {
        var response = "";
      } else{
        var response = $("#reproduction").val();
      }
      
      const network_type = replication % 4 === 0 ? 'chain' : 'network';
      const transmission_id = network_type === 'chain' ? `${replication}-${participant_index}` : `${replication}`;
      const end_time = new Date().getTime();
      const time_taken = end_time - start_time;
      var contents = {
        response,
        generation: curr_generation,
        replication,
        participant_id: dallinger.identity.participantId,
        read_multiple_versions,
        generation_size,
        read_stories,
        participant_index,
        transmission_id,
        transmission_id_hash: hashCode(transmission_id),
        network_type,
        num_stories_to_read,
        time_taken
        // bonus: running_total_pay
      }
      dallinger.createInfo(my_node_id, {
        contents: JSON.stringify(contents),
        info_type: 'Info'
      }).done(function (resp) {
        // dallinger.createAgent()
        //   .done(
        //     function(){
        //       console.log('huh')
        //     })
        //   .fail(function (rejection) {
        //     // A 403 is our signal that it's time to go to the questionnaire
        //     if (rejection.status === 403) {
        //       dallinger.allowExit();
        //       dallinger.goToPage('questionnaire');
        //     } else {
        //       dallinger.error(rejection);
        //     }
        //   });
        //dallinger.allowExit();
        //window.location = '/questionnaire' + window.location.search;
        //dallinger.goToPage('questionnaire');
        
        
        dallinger.submitAssignment();
      });
        // dallinger.goToPage('questionnaire');
        //initialSetup();
      // });
    });
  });

  function getStoryForTrial(curr_story_index,total_stories, story_list){
    if (total_stories > 1) return story_list[curr_story_index-1].contents;
    if (curr_generation === 0) return story_list[0].contents;
    const parsed_stories = story_list.map(s=>JSON.parse(s.contents));
    const relevant_index = parsed_stories.findIndex(s=>(+s.participant_index) === participant_index);
    return story_list[relevant_index].contents;  
  }

  function update_story_html(story_html,curr_story,total_stories){

    const story_str = total_stories == 1 ? 'following' : ['first','second','third','fourth'][curr_story-1];

    if (curr_generation==0 && curr_story>1){
      var h1_addition = ' (even if the same as the previous text):'
    } else if (curr_story==1){
      var h1_addition = ':'
    } else{
      var h1_addition = ' (even if similar to the previous text):'
    }

    $('#header-text').html('Read the ' +story_str+ ' text' + h1_addition);
    $('#trial-count').html('Reading page: <span>' + String(curr_story) + ' of ' + String(total_stories) + '</span>')

    const curr_story_text = getStoryForTrial(curr_story,total_stories,stories);
    const story_to_push = curr_generation === 0 ? curr_story_text : JSON.parse(curr_story_text)['response'];
    read_stories.push(story_to_push);

    const storyHTML = curr_generation===0 ? markdown.toHTML(curr_story_text) : markdown.toHTML(JSON.parse(curr_story_text)['response']);

    $("#story").html(storyHTML);
    $("p").addClass("preventcopy");
    $("#stimulus").show();
    $("#finish-reading").show();
    $('#header-text').show();
    $("#finish-reading").removeClass('disabled');
    if (curr_story==total_stories){
      $("#finish-reading").click(function(){
        $("#finish-reading").addClass('disabled')
        $('#trial_by_trial').css('margin-bottom','30px')
        $('#trial-count').html('Response page: <span>1 of 1</span>')
        // $('#trial_info_2').css('display','none')
        $("#stimulus").hide();
        $("#response-form").show();
        $("#submit-response").removeClass('disabled');
        $("#submit-response").html('Submit');
      })
      } else {
        $("#finish-reading").click(function(){
          $("#finish-reading").addClass('disabled')
          $(window).scrollTop(0);
          $("#story").html('<b>Loading story ...</b>')
          $("#finish-reading").hide()
          $('#header-text').hide()
          $('#finish-reading').off('click');
          setTimeout(function(){
            update_story_html(story_html,curr_story+1,total_stories)
          },loading_timeout)
        })
    }
  }

  // function get_story_contents(raw_stories){
  //   let story_list = [];
  //   for (let story of raw_stories){
  //     try {
  //       var response_string = JSON.parse(story)['response'];
  //     } catch {
  //       var response_string = story;
  //     }
  //     story_list.push(response_string)
  //   }
  //   return story_list
  // }

  // Create the agent.
  function initialSetup() {
    $('#finish-reading').addClass('disabled');
    my_node_id = sessionStorage.getItem('node_id');
    curr_generation = +sessionStorage.getItem('generation');
    replication = +sessionStorage.getItem('replication');
    generation_size = +sessionStorage.getItem('generation_size');
    read_multiple_versions = +sessionStorage.getItem('read_multiple_versions')
    decision_index = +sessionStorage.getItem('decision_index');
    participant_index = +sessionStorage.getItem('participant_index');
    num_stories_to_read = +sessionStorage.getItem('num_stories_to_read');
    start_time = +sessionStorage.getItem('start_time');

    if (curr_generation === 0){
      $('#response-header').html('Using only your memory of what you read, please reconstruct the article to the best of your ability');
    } else {
      if (num_stories_to_read === 1){
        $('#response-header').html("This text was another participant's attempt to describe an article. "+
        'Using only your memory of what you read, please reconstruct the original article to the best of your ability');
      } else{
        $('#response-header').html("These texts were three participants' attempts to describe the same article. "+
        "Using only your memory of what you read, please reconstruct the original article to the best of your ability.");
      }
    }
    get_info(num_stories_to_read);
  }

  function get_info(num_of_stories) {
    // Get info for node
    dallinger.getReceivedInfos(my_node_id)
      .done(function (resp) {
        stories = shuffle(resp.infos);
        update_story_html(stories, 1, num_of_stories);
        $("#response-form").hide();
      })
      .fail(function (rejection) {
        dallinger.error(rejection);
      });
  };

  initialSetup();

}());