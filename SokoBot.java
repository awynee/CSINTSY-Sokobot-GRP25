

package solver;

import java.awt.Point; // for coordinates
import java.util.HashSet;  // for set
import java.util.Set;   
import java.util.PriorityQueue;
import java.util.Comparator;

public class SokoBot {

  public String solveSokobanPuzzle(int width, int height, char[][] mapData, char[][] itemsData) 
  {
		
		Point player = null;
		Set<Point> boxes = new HashSet();
		Set<Point> targets = new HashSet();
		
		//scanning of the map and marking points
		for(int i = 0; i < height; i++){
			for(int j = 0; j < width; j++) {
				if(mapData[i][j] == '.') //targets
					targets.add(new Point(j,i));
				if(itemsData[i][j] == '$') //boxes
					boxes.add(new Point(j,i));
				if(itemsData[i][j] == '@') //player
					player = new Point(j,i);
			}
		}
		
		class State { // current game state
			Point player;
			Set<Point> boxes;
			String path;
			int cost; // cost so far
			int heuristic; // heuristic
			int total; // total cost = cost + heuristic
			
			State(Point player, Set<Point> boxes, String path, int cost, int heuristic) {
				this.player = player;
				this.boxes = boxes;
				this.path = path;
				this.cost = cost;
				this.heuristic = heuristic;
				this.total = cost + heuristic;
			}

		}
		
		PriorityQueue<State> open = new PriorityQueue<>(Comparator.comparingInt(s -> s.total));		
		open.add(new State(player, boxes, "", 0, heuristic(boxes, targets))); //starting state of the map 
		
		Set<String> visited = new HashSet<>(); //explored positions
		
		long startTime = System.currentTimeMillis();

        // A* loop start
        while (!open.isEmpty()) {
            // check 15-second timeout
            if (System.currentTimeMillis() - startTime > 15000)
                break;

            // get lowest total cost state
            State current = open.poll();

            // create a unique identifier for visited check
            String key = current.player.x + "," + current.player.y + current.boxes.toString();
            if (visited.contains(key))
                continue;
            visited.add(key);

            // check goal condition
            if (targets.containsAll(current.boxes))
				return current.path;

            // to do: need to generate new states (where the player moves and pushes boxes)
		}
		return "";				
	}
	private int manhattan(Point a, Point b) { // formula for heuristic
		return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
	}
		
	private int heuristic(Set<Point> boxes, Set<Point> targets) {
		int total = 0;
		for (Point box : boxes) {
			int best = Integer.MAX_VALUE;
			for (Point goal : targets) {
				int d = Math.abs(box.x - goal.x) + Math.abs(box.y - goal.y);
				if (d < best) best = d;
			}
			total += best;
		}
		return total;
	}
	
}
